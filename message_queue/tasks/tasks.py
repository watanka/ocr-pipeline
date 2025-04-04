from common.schema import DetectionResponse, BatchRecognitionRequest, OCRResponse, BatchRecognitionResponse
from typing import Dict, Any



import logging
logger = logging.getLogger("message_queue.tasks.std_task")

# JSON 객체를 나타내는 타입
JsonDict = Dict[str, Any]

def create_detection_message(detection_result: DetectionResponse) -> JsonDict:    
    """STD 결과를 메시지 큐용 메시지로 변환 (PIPELINE -> STR)"""
    recognition_request = BatchRecognitionRequest(
        request_id=detection_result.request_id,
        regions=detection_result.regions
    )

    return recognition_request.model_dump(mode="json")

def create_recognition_message(recognition_result: BatchRecognitionResponse) -> JsonDict:
    """STR 결과를 메시지 큐용 메시지로 변환 (STR -> PIPELINE)"""
    return recognition_result.model_dump(mode="json")

