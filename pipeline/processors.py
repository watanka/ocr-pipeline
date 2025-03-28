import cv2
import numpy as np
import base64
import httpx
from typing import Dict, List, Any
import logging
import uuid
import os
from common.schema import DetectionResponse, DetectionRequest, BatchRecognitionRequest, BatchRecognitionResponse
from fastapi import UploadFile
from common.logger import setup_logger, LOG_FORMATS
from message_queue.bucket import BatchBucket


# 로거 설정
logger = setup_logger(
    'pipeline.processors',
    format_string=LOG_FORMATS['DETAILED']
)

# 서비스 URL
DETECTION_URL = os.getenv("DETECTION_URL", "http://std-detection:8000/detection")
RECOGNITION_URL = os.getenv("RECOGNITION_URL", "http://str-recognition:8001/recognition")
BATCH_RECOGNITION_URL = os.getenv("BATCH_RECOGNITION_URL", "http://str-recognition:8001/batch_recognition")

# 타임아웃 설정 (초)
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))


async def convert_to_base64(file: UploadFile) -> str:
    # 파일 내용을 바이트로 읽기
    contents = await file.read()
    
    # 바이트를 base64로 인코딩
    base64_encoded = base64.b64encode(contents).decode('utf-8')
    
    # 파일 포인터를 처음으로 되돌리기 (다시 읽을 수 있도록)
    await file.seek(0)
    
    return base64_encoded


async def convert2image(file: UploadFile):
    # 파일 내용을 바이트로 읽기
    contents = await file.read()
    
    # 바이트 데이터를 numpy array로 변환
    nparr = np.frombuffer(contents, np.uint8)
    
    # numpy array를 이미지로 디코딩
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 파일 포인터를 처음으로 되돌리기 (다시 읽을 수 있도록)
    await file.seek(0)
    
    return image

def base64_to_image(base64_string: str) -> np.ndarray:
    try:
        # 디코딩
        image_data = base64.b64decode(base64_string)

        if not image_data:
            return None

        # NumPy 배열 변환
        np_arr = np.frombuffer(image_data, np.uint8)

        if np_arr.size == 0:
            return None

        # OpenCV 디코딩
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return None

        return image

    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        raise

async def process_std(request_id: str, encoded_image: str, file_name: str) -> DetectionResponse:
    try:
        request_data = DetectionRequest(
            request_id=request_id,
            file_name=file_name,
            image=encoded_image
        )
        
        # API 호출
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                DETECTION_URL,
                json=request_data.model_dump()
            )
            response.raise_for_status()
            result: DetectionResponse = DetectionResponse.model_validate(response.json()) 
            
            return result
            
    except Exception as e:
        logger.error(f"STD 처리 중 오류 발생: {str(e)}")
        raise

async def process_str(detection_result: DetectionResponse) -> BatchRecognitionResponse:
    try:
        batch_request = BatchRecognitionRequest(
            request_id=detection_result.request_id,
            regions=detection_result.regions
        )
        # API 호출
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(BATCH_RECOGNITION_URL, json=batch_request.model_dump())
            response.raise_for_status()
            result: BatchRecognitionResponse = BatchRecognitionResponse.model_validate(response.json())
            
            return result
            
    except Exception as e:
        logger.error(f"STR 처리 중 오류 발생: {str(e)}")
        raise


async def process_str_with_bucket(bucket: BatchBucket, detection_result: DetectionResponse) -> BatchRecognitionResponse:
    
    try:
        batch_request = BatchRecognitionRequest(
            request_id=detection_result.request_id,
            regions=detection_result.regions
        )
        # API 호출
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(BATCH_RECOGNITION_URL, json=batch_request.model_dump())
            response.raise_for_status()
            result: BatchRecognitionResponse = BatchRecognitionResponse.model_validate(response.json())
            
        await bucket.add(result)
        
        return result
        
            
    except Exception as e:
        logger.error(f"STR 처리 중 오류 발생: {str(e)}")
        raise

def create_result_image(image: np.ndarray, batch_recognition_result: BatchRecognitionResponse) -> np.ndarray:
    try:
        # 원본 이미지 복사
        result_image = image.copy()
        
        # 각 결과에 대해 박스와 텍스트 표시
        for result in batch_recognition_result.results:
            logger.info(f"result: {result}")
            
            if not result.bbox:
                continue
                
            
            # 박스 그리기
            pts = np.array(result.bbox, np.int32)
            cv2.polylines(result_image, [pts], True, (0, 255, 0), 2)
            
            # 텍스트 표시
            if result.text:
                x, y = result.bbox[0]
                cv2.putText(result_image, result.text, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result_image
        
    except Exception as e:
        logger.error(f"결과 이미지 생성 중 오류 발생: {str(e)}")
        raise 