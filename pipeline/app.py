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
def pipeline(req: DetectionRequest) -> list[RecognitionResponseV2]:
    # detection에서 글자 탐지 후, 탐지된 글자 영역을 크롭한 이미지를 저장
    detection_response = requests.post(f'http://detection:8000/detection', json=req.model_dump()) 
    # recognition 처리는 저장된 파일로부터 수행
    detection_result = detection_response.json()
    recognition_response = requests.post(f'http://recognition:8001/recognition', json=detection_result)
    recognition_result = recognition_response.json()    
    
    return recognition_result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)




