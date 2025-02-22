from pydantic import BaseModel


class RecognitionRequestV1(BaseModel):
    file_name: str
    crop_file_name: str

class RecognitionRequestV2(BaseModel):
    id: str
    file_name: str
    image: str # base64 encoded


class RecognitionResponseV2(BaseModel):
    id: str
    file_name: str
    text: str
    confidence: float
