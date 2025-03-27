from pydantic import BaseModel, Field
from typing import Optional, List
from .recognition import RecognitionResult

class OCRResponse(BaseModel):
    """최종 OCR 결과 모델"""
    id: str = Field(..., description="요청 ID")
    file_name: str = Field(..., description="파일 이름")
    regions: List[RecognitionResult] = Field(..., description="감지 및 인식 결과")
    processing_time: float = Field(0.0, description="총 처리 시간 (초)")
    visualized_image: Optional[str] = Field(None, description="텍스트가 표시된 결과 이미지 (Base64)")


class PingResponse(BaseModel):
    """서비스 상태 응답 모델"""
    status: str = Field(..., description="서비스 상태")
