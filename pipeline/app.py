import os
import uuid
import logging
import traceback
import sys
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import uvicorn
import json
import asyncio
from functools import partial
from datetime import datetime
from message_queue.bucket import BatchBucket, STDBucket, STRBucket
from message_queue.monitor import BucketMonitor, STDBucketMonitor, STRBucketMonitor
from message_queue.factory import MessageQueueFactory
from message_queue.base import MessageQueue
from message_queue.tasks import create_detection_message
from processors import (process_std, 
                        process_str, 
                        process_str_with_bucket,
                        create_result_image, 
                        convert2image, 
                        convert_to_base64,
                        )

from file_utils import save_result
from common.schema import OCRResponse, PingResponse, DetectionResponse, BatchRecognitionResponse, BatchRecognitionRequest, RecognitionResult
from common.logger import setup_logger, LOG_FORMATS

# 모니터링 모듈 임포트
sys.path.append('/app/monitoring')
from monitoring import get_monitor

# 메시지 큐 모듈 임포트
sys.path.append('/app/message_queue')

# 로거 설정
logger = setup_logger(
    'pipeline',
    format_string=LOG_FORMATS['DETAILED']
)

# 모니터 초기화
monitor = get_monitor("ocr-pipeline")

# 메시지 큐 설정
QUEUE_TYPE = os.getenv("QUEUE_TYPE", "rabbitmq")  # 기본값은 rabbitmq
QUEUE_URL = os.getenv("QUEUE_URL", "amqp://admin:admin@rabbitmq:5672/%2F")

# 메시지 큐 초기화
message_queue = MessageQueueFactory.create(
    queue_type=QUEUE_TYPE,
    url=QUEUE_URL
)

# FastAPI 앱 설정
app = FastAPI(title="OCR Pipeline API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 메시지 큐 클라이언트
app.mq: Optional[MessageQueue] = None

# 서비스 URL
DETECTION_URL = "http://std-detection:8000/detection"
RECOGNITION_URL = "http://str-recognition:8001/recognition"
BATCH_RECOGNITION_URL = "http://str-recognition:8001/batch_recognition"


@app.get("/ping", response_model=PingResponse)
async def ping():
    logger.info("Ping request received")
    return PingResponse(
        status="ok", 
        message="Pipeline service is running"
    )

@app.get("/health")
async def health_check():
    """
    서비스 상태를 확인합니다.
    """
    return {"status": "ok"}

@app.get("/metrics")
async def get_metrics():
    """
    서비스 성능 지표를 반환합니다.
    """
    summary = monitor.get_summary()
    return {
        "service": "pipeline",
        "metrics": summary
    }

def preprocess_std_message(messages: List[DetectionResponse]):
    """std버켓에 넣기 전 처리"""
    return messages.regions

def postprocess_std_message(messages: List[DetectionResponse]):
    """std버켓에서 꺼낼 때 처리"""
    return DetectionResponse(
        request_id=str(uuid.uuid4())[:10],
        regions=messages,
        result_image=None,
        total_regions=len(messages)
    )

@app.on_event("startup")
async def startup_event():
    logger.info("OCR Pipeline API started")
    logger.info(f"Detection service URL: {DETECTION_URL}")
    logger.info(f"Recognition service URL: {RECOGNITION_URL}")
    logger.info(f"Batch recognition service URL: {BATCH_RECOGNITION_URL}")
    logger.info(f"Using message queue: {QUEUE_TYPE}")
    logger.info("Ready to process requests")
    
    # 메시지 큐 연결
    # await message_queue.connect()

    app.std_bucket = STDBucket(max_batch_size=10, wait_time=1)
    app.str_bucket = STRBucket()


    app.std_bucket_monitor = STDBucketMonitor(
        bucket=app.std_bucket,
        interval = 10.0
    )
    app.result_queue = asyncio.Queue()
    app.str_bucket_monitor = STRBucketMonitor(bucket=app.str_bucket, result_queue=app.result_queue)

    
    asyncio.create_task(app.std_bucket_monitor.monitor(process_message = postprocess_std_message, 
                                                       callback=partial(process_str_with_bucket, app.str_bucket)))
    asyncio.create_task(app.str_bucket_monitor.monitor())
        

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Pipeline API")
    # 메시지 큐 연결 해제
    # await message_queue.disconnect()
    logger.info("Service stopped")


@app.post("/ocr")
async def process_ocr(file: UploadFile = File(...), 
                      background_tasks: BackgroundTasks = BackgroundTasks()) -> OCRResponse:
    """
    OCR 파이프라인 실행
    """
    try:
        # 요청 정보 로깅
        logger.info(f"OCR request received for file: {file.filename}")
        request_id = str(uuid.uuid4())[:10] + datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename.split('.')[0]
        
        # 모니터링 시작
        monitor.start_timer("ocr_total")
        
        # 이미지 로드
        image = await convert2image(file)
        if image is None:
            raise HTTPException(status_code=400, detail="이미지를 로드할 수 없습니다.")
        img_base64 = await convert_to_base64(file)

        # STD 처리
        monitor.start_timer("std_detection")
        std_result: DetectionResponse = await process_std(request_id, img_base64, file.filename)
        std_time = monitor.stop_timer("std_detection")
        logger.info(f"STD completed in {std_time:.4f}s")

        # STR 처리
        monitor.start_timer("str_recognition")
        str_result: BatchRecognitionResponse = await process_str(std_result)
        str_time = monitor.stop_timer("str_recognition")
        logger.info(f"STR completed in {str_time:.4f}s")
        
        # 결과 이미지 생성
        result_image = create_result_image(image, str_result)

        # 총 소요 시간 기록
        total_time = monitor.stop_timer("ocr_total")
        logger.info(f"OCR total processing time: {total_time:.4f}s")

        # 리소스 사용량 기록
        resources = monitor.get_system_resources()
        logger.info(f"Resource usage: {resources}")

        background_tasks.add_task(save_result, image, file.filename, request_id, std_result, str_result)

        return OCRResponse(
            id=request_id,
            file_name=file.filename,
            regions=str_result.results,
            processing_time=total_time
        )
        
    except Exception as e:
        logger.error(f"OCR 처리 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/batch_ocr", response_model=OCRResponse)
async def process_ocr_with_queue(file: UploadFile = File(...), 
                                background_tasks: BackgroundTasks = BackgroundTasks()) -> OCRResponse:
    request_id = str(uuid.uuid4())[:10] + datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename.split('.')[0]
    try:
        monitor.start_timer("pipeline_total")
        logger.info(f"OCR request received for file: {file.filename}")

        image = await convert2image(file)
        if image is None:
            raise HTTPException(status_code=400, detail="이미지를 로드할 수 없습니다.")
        img_base64 = await convert_to_base64(file)
        
        # STD 처리
        monitor.start_timer("std_detection")
        std_result: DetectionResponse = await process_std(request_id, img_base64, file.filename)
        det_time = monitor.stop_timer("std_detection")
        logger.warning(f"STD completed in {det_time:.4f}s, found {len(std_result.regions)} regions")
        monitor.record_metric("detected_regions", len(std_result.regions))
        
        # STD 결과 큐에 발행
        # detection_message = create_detection_message(detection_result)
        # await message_queue.publish("std_results", detection_message)


        # STD 결과 버켓에 저장
        await app.std_bucket.add(std_result)
        logger.info(f"{len(std_result.regions)} STD result added to bucket")

        # STR 처리
        monitor.start_timer("str_recognition")
        # str_result: BatchRecognitionResponse = await message_queue.consume(
        #     "str_results",
        #     timeout=30.0,
        #     filter_fn=lambda x: x
        # )

        str_result = await app.result_queue.get()
        str_result: BatchRecognitionResponse = BatchRecognitionResponse.model_validate(dict(str_result))

        str_time = monitor.stop_timer("str_recognition")
        logger.info(f"STR completed in {str_time:.4f}s")

        total_time = monitor.stop_timer("pipeline_total")
        logger.info(f"Batch OCR total processing time: {total_time:.4f}s")

        # 리소스 사용량 기록
        resources = monitor.get_system_resources()
        logger.info(f"Resource usage: {resources}")


        background_tasks.add_task(save_result, image, file.filename, request_id, std_result, str_result)

        return OCRResponse(
            id=request_id,
            file_name=file.filename,
            regions=str_result.results,
            processing_time=total_time
        )
        
    except Exception as e:
        logger.error(f"Error in message queue OCR pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        monitor.stop_timer("pipeline_total")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=True)




