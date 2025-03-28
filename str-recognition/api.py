import os
import time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import traceback
import sys

import asyncio
from typing import Optional
from message_queue.factory import MessageQueueFactory
from message_queue.message_process.hybrid import HybridStrategy
from message_queue.tasks.tasks import create_recognition_message
from message_queue.base import MessageQueue
from message_queue.bucket import BatchBucket
from message_queue.monitor import BucketMonitor

# 모니터링 모듈 임포트
sys.path.append('/app/monitoring')
from monitoring import get_monitor

from model import Model
from utils import CTCLabelConverter, AttnLabelConverter, load_model_dict
from common.schema import (
    BatchRecognitionRequest, 
    RecognitionResult, 
    BatchRecognitionResponse
)
from processor import convert2image, prepare_batch_tensor, post_process_text
from common.logger import setup_logger, LOG_FORMATS

# RabbitMQ 연결 설정
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://admin:admin@rabbitmq:5672/%2F")

# 로거 설정
logger = setup_logger(
    'str_recognition',
    format_string=LOG_FORMATS['DETAILED']
)

# 모니터 초기화
monitor = get_monitor("str-recognition")

# FastAPI 앱 생성
app = FastAPI(title="STR Recognition API")

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

# CUDA 정보 출력
if torch.cuda.is_available():
    logger.info(f"CUDA is available, device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
else:
    logger.info("CUDA is not available")

# 모델 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# 모델 옵션 설정
opt = {
    'Transformation': 'TPS',
    'FeatureExtraction': 'ResNet',
    'SequenceModeling': 'BiLSTM',
    'Prediction': 'Attn',
    'num_fiducial': 20,
    'imgH': 32,
    'imgW': 100,
    'character': '0123456789abcdefghijklmnopqrstuvwxyz',
    'input_channel': 1,
    'output_channel': 512,
    'hidden_size': 256,
    'num_fiducial': 20,
    'batch_max_length': 25,
    'rgb': True
}

# 변환기 초기화
if 'CTC' in opt['Prediction']:
    converter = CTCLabelConverter(opt['character'])
else:
    converter = AttnLabelConverter(opt['character'])
opt['num_class'] = len(converter.character)

# 모델 초기화
try:
    logger.info("Initializing text recognition model")
    start_time = time.time()
    
    model = Model(opt)
    model = model.to(device)
    
    # 모델 가중치 파일 경로
    model_path = 'saved_models/TPS-ResNet-BiLSTM-Attn.pth'
    model_abs_path = os.path.abspath(model_path)
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Absolute path: {model_abs_path}")
    logger.info(f"Model file exists: {os.path.exists(model_abs_path)}")
    
    # 모델 가중치 로드
    model = load_model_dict(model, model_path)
    model.eval()
    
    # 모델 로드 시간 기록
    model_load_time = time.time() - start_time
    monitor.record_metric("model_load_time", model_load_time)
    logger.info(f"Model loaded successfully in {model_load_time:.2f}s")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    logger.error(traceback.format_exc())
    logger.error(f"Current working directory: {os.getcwd()}")
    logger.error(f"Files in saved_models directory: {os.listdir('saved_models') if os.path.exists('saved_models') else 'Directory not found'}")
    raise


@app.post("/batch_recognition", response_model=BatchRecognitionResponse)
async def batch_recognize_text(request: BatchRecognitionRequest):
    """
    배치로 텍스트 영역에서 텍스트 인식
    """
    try:
        # 모니터링 시작
        monitor.start_timer("batch_recognition")
        
        # 배치 요청 정보 로깅
        regions = request.regions
        logger.info(f"Batch recognition request received: {len(regions)} regions")
        
        # 1. 이미지 전처리 단계
        monitor.start_timer("preprocessing")
        processed_images = []
        valid_indices = []
        bboxes = {}
        
        for i, region in enumerate(regions):
            try:
                img = convert2image(region)
                processed_images.append(img)
                valid_indices.append(i)
                bboxes[region.request_id] = region.bbox
                
            except Exception as e:
                logger.error(f"Error preprocessing region {i} (ID: {region.request_id}): {str(e)}")
                continue
        
        preprocess_time = monitor.stop_timer("preprocessing")
        logger.info(f"Preprocessing completed in {preprocess_time:.4f}s: {len(processed_images)} valid images")
        
        # 2. 모델 추론 단계
        monitor.start_timer("model_inference")
        results = []
        
        if processed_images:
            try:
                # 배치 텐서 준비
                batch_tensor = prepare_batch_tensor(processed_images)
                
                # 모델 추론
                with torch.no_grad():
                    preds = model(batch_tensor)
                
                # 결과 디코딩
                if 'CTC' in opt['Prediction']:
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, torch.IntTensor([preds.size(1)]))
                else:
                    preds_index = preds.argmax(dim=2)
                    # 각 이미지별로 길이 정보 생성
                    lengths = torch.IntTensor([preds.size(1)] * preds.size(0))
                    preds_str = converter.decode(preds_index, lengths)
                
                # logger.info(f"preds_str: {preds_str}")
                # 결과 처리
                for i, (idx, text) in enumerate(zip(valid_indices, preds_str)):
                    region = regions[idx]
                    processed_text = post_process_text(text)
                    
                    results.append(RecognitionResult(
                        id=region.request_id,
                        text=processed_text,
                        bbox=bboxes[region.request_id]
                    ))
                    
                    # logger.info(f"Region {idx} (ID: {region.id}) recognized: '{processed_text}'")
                    
                    # 메트릭 기록
                    monitor.record_metric("text_length", len(processed_text))
                    
            except Exception as e:
                logger.error(f"Error in model inference: {str(e)}")
                logger.error(traceback.format_exc())
                # 모델 추론 실패 시 빈 결과 반환
                for idx in valid_indices:
                    region = regions[idx]
                    results.append(RecognitionResult(
                        id=region.request_id,
                        text="",
                        error="Model inference failed",
                        bbox=bboxes[region.request_id]
                    ))
        
        inference_time = monitor.stop_timer("model_inference")
        logger.info(f"Model inference completed in {inference_time:.4f}s")
        
        # 3. 실패한 영역에 대한 처리
        for i, region in enumerate(regions):
            if i not in valid_indices:
                results.append(RecognitionResult(
                    id=region.request_id,
                    text="",
                    error="Preprocessing failed",
                    bbox=region.bbox
                ))
        
        # 총 처리 시간 및 메트릭 기록
        total_time = monitor.stop_timer("batch_recognition")
        recognized_count = sum(1 for r in results if r.text)
        
        logger.info(f"Batch recognition completed in {total_time:.4f}s: {recognized_count}/{len(regions)} regions recognized")
        
        # 메트릭 기록
        monitor.record_metric("batch_size", len(regions))
        monitor.record_metric("recognition_success_rate", recognized_count / len(regions) if len(regions) > 0 else 0)
        monitor.record_metric("preprocessing_time", preprocess_time)
        monitor.record_metric("inference_time", inference_time)
        
        return BatchRecognitionResponse(results=results, request_id=request.request_id)
        
    except Exception as e:
        logger.error(f"Error processing batch recognition request: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/metrics")
async def get_metrics():
    """
    서비스 성능 지표를 반환합니다.
    """
    summary = monitor.get_summary()
    return {
        "service": "str-recognition",
        "metrics": summary
    }


async def str_api_call(message):
    """
    메시지 큐에서 받은 STD 결과를 처리합니다.
    """
    try:
        # 메시지 데이터 파싱
        logger.info(f'from process_std_message: {message}')
        # logger.info(f"Received STD message: {message['request_id']}")
        # logger.info(f"Received STD message: {message['regions'][0]}")
        # BatchRecognitionRequest 형식으로 변환
        regions = []
        for msg in message:
            regions.extend(msg['regions'])

        request = BatchRecognitionRequest(
            regions=regions
        )
        
        # batch_recognize_text 엔드포인트 호출
        response: BatchRecognitionResponse = await batch_recognize_text(request)
        
        logger.info(f"Processed STD message: {len(response.results)} regions recognized")

        # 결과 메시지 생성
        recognition_message = create_recognition_message(response)
        # logger.info(f"Publishing result message with request_id: {recognition_message.get('request_id')}")
        # await app.mq.publish("str_results", recognition_message)
        
        return response
    except Exception as e:
        logger.error(f"Failed during startup: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        raise

        
 
# 애플리케이션 시작 로그
@app.on_event("startup")
async def startup_event():
    logger.info("STR Recognition API startup event started")
    try:
        # 메시지 큐 클라이언트 생성
        # logger.info("Creating message queue client...")
        # logger.info(f"Using RabbitMQ URL: {RABBITMQ_URL}")
        # # app.mq = MessageQueueFactory.create("rabbitmq", url=RABBITMQ_URL)
        # logger.info("Message queue client created successfully")
        
        # # 메시지 큐 연결
        # logger.info("Attempting to connect to RabbitMQ...")
        # await app.mq.connect()
        # logger.info("Successfully connected to RabbitMQ")
        
        # STD 결과 구독 시작 (백그라운드 태스크로 실행)
        # asyncio.create_task(subscribe_to_std_results())

        # bucket = BatchBucket(max_batch_size=100, wait_time=10)
        # monitor = BucketMonitor(bucket, 10)

        # await app.mq.pour(bucket, 'std_results')
        # asyncio.create_task(monitor.monitor(str_api_call))
        # logger.info("Started subscription task for STD results queue")
        
        logger.info("STR Recognition API startup completed successfully")
    except Exception as e:
        logger.error(f"Failed during startup: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        raise

async def process_std_message(message):
    """
    메시지 큐에서 받은 STD 결과를 처리합니다.
    """
    try:
        # 메시지 데이터 파싱
        logger.info(f'from process_std_message: {message}')
        # logger.info(f"Received STD message: {message['request_id']}")
        # logger.info(f"Received STD message: {message['regions'][0]}")
        # BatchRecognitionRequest 형식으로 변환
        regions = []
        for msg in message:
            regions.extend(msg['regions'])

        request = BatchRecognitionRequest(
            regions=regions
        )
        
        # batch_recognize_text 엔드포인트 호출
        response: BatchRecognitionResponse = await batch_recognize_text(request)
        
        logger.info(f"Processed STD message: {len(response.results)} regions recognized")

        # 결과 메시지 생성
        recognition_message = create_recognition_message(response)
        logger.info(f"Publishing result message with request_id: {recognition_message.get('request_id')}")
        await app.mq.publish("str_results", recognition_message)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing STD message: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def subscribe_to_std_results():
    """STD 결과 구독"""
    try:
        logger.info("Setting up subscription to std_results queue...")
        strategy = HybridStrategy(max_batch_size=100, wait_time=10)
        await app.mq.handle_str_request("std_results", 
                                        process_std_message, 
                                        strategy)
        logger.info("Successfully subscribed to std_results queue")
    except Exception as e:
        logger.error(f"Failed to subscribe to STD results: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# 애플리케이션 종료 시 로깅
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("STR Recognition API shutting down")
    logger.info("Service stopped")

    # 메시지 큐 연결 해제
    if app.mq:
        await app.mq.disconnect()
        logger.info("Disconnected from message queue")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True) 