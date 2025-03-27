import time
import cv2
import numpy as np

from fastapi import FastAPI, HTTPException

import base64
import logging
import traceback
import sys

# 모니터링 모듈 임포트
sys.path.append('/app/monitoring')
from monitoring import get_monitor
from common.schema import (
    DetectionRequest, 
    SingleDetection, 
    DetectionResponse
)

from craft import CRAFT
from test import test_net
from file_utils import byte2image, image2base64, crop_text_region
from detection_utils import initialize_model
from common.logger import setup_logger, LOG_FORMATS

# 로거 설정
logger = setup_logger(
    'std_detection',
    format_string=LOG_FORMATS['DETAILED']
)

# 모니터 초기화
monitor = get_monitor("std-detection")

app = FastAPI(title="STD Detection API")


# CRAFT 모델 설정
try:
    logger.info("Initializing CRAFT model")
    start_time = time.time()
    
    net = CRAFT()

    # 모델 파일 경로
    model_path = 'weights/craft_mlt_25k.pth'
    logger.info(f"Loading model from: {model_path}")
    
    # 모델 초기화
    net = initialize_model(net, model_path)
    
    # 모델 로드 시간 기록
    model_load_time = time.time() - start_time
    monitor.record_metric("model_load_time", model_load_time)
    logger.info(f"Model loaded successfully in {model_load_time:.2f}s")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    logger.error(traceback.format_exc())
    raise

@app.post("/detection", response_model=DetectionResponse)
async def detect_text(request: DetectionRequest):
    """
    이미지에서 텍스트 영역 감지
    """
    try:
        # 전체 처리 시간 측정 시작
        monitor.start_timer("detection_total")
        # 이미지 디코딩
        monitor.start_timer("image_preprocessing")
        try:
            img, height, width = byte2image(request.image)
            # 전처리 완료 측정
            preproc_time = monitor.stop_timer("image_preprocessing")
            logger.info(f"Image preprocessing completed in {preproc_time:.4f}s")

        except Exception as e:
            monitor.stop_timer("image_preprocessing")
            raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")
        
        # 텍스트 영역 감지
        monitor.start_timer("text_detection")
        try:
            # 모델을 통한 텍스트 감지
            # test_net 함수는 bboxes, polys, score_text를 반환함
            bboxes, polys, score_text = test_net(net, img, 0.7, 0.4, 0.4, False, False, None)
            
            # TextRegion 객체 리스트로 변환
            text_regions = []
            for i, (box, poly) in enumerate(zip(bboxes, polys)):
                # 박스 좌표를 파이썬 기본 타입으로 변환
                box_points = box.astype(np.int32).tolist()
                
                # 텍스트 영역 주변 이미지 자르기
                x_min = int(min([p[0] for p in box]))
                y_min = int(min([p[1] for p in box]))
                x_max = int(max([p[0] for p in box]))
                y_max = int(max([p[1] for p in box]))
                
                
                
                # 텍스트 영역 이미지 잘라내기
                region_img = crop_text_region(img, x_min, y_min, x_max, y_max)
                region_img_base64 = image2base64(region_img)
                
                # TextRegion 객체 생성
                region_id = f"{request.file_name.split('.')[0]}_{i}"
                text_region = SingleDetection(
                    id=region_id,
                    image=region_img_base64,
                    bbox=box_points,
                    polygon=[[int(point) for point in p.tolist()] for p in poly],
                    confidence=0
                )
                
                text_regions.append(text_region)
            
            # 감지된 영역 정보 수집
            detection_time = monitor.stop_timer("text_detection")
            monitor.increment_counter("text_regions_count", len(text_regions))
            
            logger.info(f"Text detection completed in {detection_time:.4f}s, found {len(text_regions)} regions")
        except Exception as e:
            monitor.stop_timer("text_detection")
            raise HTTPException(status_code=500, detail=f"Text detection failed: {str(e)}")
        
        # # 결과 이미지 생성 (선택 사항)
        # monitor.start_timer("result_image")
        # try:    
        #     # 원본 이미지에 텍스트 영역 표시
        #     result_img = draw_detection_boxes(img.copy(), bboxes)
            
        #     # 결과 이미지를 Base64로 인코딩
        #     is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        #     if not is_success:
        #         raise ValueError("Failed to encode result image")
                
        #     result_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
        #     result_image_time = monitor.stop_timer("result_image")
        #     logger.info(f"Result image generation completed in {result_image_time:.4f}s")
        # except Exception as e:
        #     monitor.stop_timer("result_image")
        #     result_image_base64 = None
        
        # 응답 준비
        response = DetectionResponse(
            id = request.id,
            regions=text_regions,
            result_image=None,
            total_regions=len(text_regions)
        )
        
        # 총 처리 시간 측정 종료
        total_time = monitor.stop_timer("detection_total")
        
        # 이미지 처리 요약 정보 로깅
        monitor.log_image_processing(
            image_id=request.file_name,
            process_type="detection",
            elapsed_time=total_time,
            boxes_count=len(text_regions),
            extra_info={
                "image_size": len(region_img_base64),
                "image_dimensions": f"{width}x{height}"
            }
        )
        
        # 메트릭 기록
        monitor.record_metric("image_size", len(region_img_base64))
        
        logger.info(f"Total detection time: {total_time:.4f}s for {len(text_regions)} regions")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing detection request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        "service": "std-detection",
        "metrics": summary
    }

@app.on_event("startup")
async def startup_event():
    logger.info("STD Detection API started")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("STD Detection API shutting down")
    logger.info("Service stopped")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 