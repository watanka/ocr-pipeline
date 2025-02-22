from fastapi import FastAPI 
from pydantic import BaseModel
from kafka import KafkaProducer
import json
import requests
import yaml


def load_config(config_path: str = "configs/config.yml") -> dict:
    """
    YAML 설정 파일을 읽어서 딕셔너리로 반환합니다.
    
    Args:
        config_path (str): YAML 설정 파일 경로
        
    Returns:
        dict: 설정 값들을 담은 딕셔너리
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML 파일 파싱 중 오류 발생: {e}")

config = load_config()
pipeline_config = config['pipeline']


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
    detection_response = requests.post(f'{pipeline_config["detection_server"]}', json=req.model_dump()) 
    # recognition 처리는 저장된 파일로부터 수행
    detection_result = detection_response.json()
    recognition_response = requests.post(f'{pipeline_config["recognition_server"]}', json=detection_result)
    recognition_result = recognition_response.json()    
    
    return recognition_result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)




