from fastapi import FastAPI 
from pydantic import BaseModel
import requests


class DetectionRequest(BaseModel):
    file_name: str
    image: str # base64 encoded


class RecognitionResponseV2(BaseModel):
    id: str
    file_name: str
    text: str
    confidence: float


app = FastAPI()

@app.post("/pipeline")
def pipeline(req: DetectionRequest) -> RecognitionResponseV2:
    # detection에서 글자 탐지 후, 탐지된 글자 영역을 크롭한 이미지를 저장
    detection_response = requests.post(f'http://detection:8000/detection', json=req.detection_request) 
    # recognition 처리는 저장된 파일로부터 수행
    recognition_response = requests.post(f'http://recognition:8001/recognition', json=req.recognition_request)

    return recognition_response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)




