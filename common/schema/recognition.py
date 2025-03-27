from pydantic import BaseModel, Field
from typing import Optional, List
from .detection import SingleDetection

class RecognitionRequest(BaseModel):
    """
    단일 이미지에 대한 텍스트 인식 요청 모델
    """
    id: Optional[str] = Field(default="unknown", description="텍스트 영역 ID")
    image: str = Field(..., description="Base64로 인코딩된 이미지 데이터")


class RecognitionResult(BaseModel):
    """텍스트 인식 결과 모델"""
    id: str = Field(..., description="영역 ID")
    text: str = Field(..., description="인식된 텍스트")
    error: Optional[str] = Field(None, description="오류 메시지 (있는 경우)")
    bbox: List[List[int]] = Field(..., description="텍스트 박스 좌표 (폴리곤)")


class BatchRecognitionRequest(BaseModel):
    """배치 텍스트 인식 요청 모델"""
    request_id: Optional[str] = Field(None, description="요청 ID")
    regions: List[SingleDetection] = Field(..., description="인식할 텍스트 영역 목록")

class BatchRecognitionResponse(BaseModel):
    """배치 텍스트 인식 결과 모델"""
    request_id: Optional[str] = Field(None, description="요청 ID")
    results: List[RecognitionResult] = Field(..., description="인식 결과 목록")
