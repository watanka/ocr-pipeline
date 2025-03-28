from pydantic import BaseModel, Field
from typing import Optional, List


class DetectionRequest(BaseModel):
    """
    텍스트 감지 요청 모델
    """
    request_id: str = Field(..., description="request_id")
    file_name: Optional[str] = Field(default="image.jpg", description="이미지 파일명")
    image: str = Field(..., description="Base64로 인코딩된 이미지 데이터")


class SingleDetection(BaseModel):
    """
    감지된 텍스트 영역 모델
    """
    request_id: str = Field(..., description="{request_id}_{file_name}_{idx}")
    image: str = Field(..., description="Base64로 인코딩된 크롭된 이미지")
    bbox: List[List[int]] = Field(..., description="텍스트 박스 좌표 (폴리곤)")
    polygon: Optional[List[List[int]]] = Field(None, description="텍스트 박스 좌표 (폴리곤)")
    confidence: Optional[float] = Field(default=0, description="감지 신뢰도")

class DetectionResponse(BaseModel):
    """
    텍스트 감지 응답 모델
    """
    request_id: str = Field(..., description="요청 ID")
    regions: List[SingleDetection] = Field(default=[], description="감지된 텍스트 영역 목록")
    result_image: Optional[str] = Field(None, description="텍스트 박스가 표시된 결과 이미지 (Base64)")
    total_regions: int = Field(..., description="감지된 텍스트 영역 수") 
