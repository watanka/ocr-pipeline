from .base import OCRResponse, PingResponse
from .detection import DetectionRequest, DetectionResponse, SingleDetection
from .recognition import RecognitionRequest, RecognitionResult, BatchRecognitionRequest, BatchRecognitionResponse

__all__ = ["OCRResponse", "PingResponse", 
           "DetectionRequest", "DetectionResponse", "SingleDetection", 
           "RecognitionRequest", "RecognitionResult", "BatchRecognitionRequest", "BatchRecognitionResponse"]

