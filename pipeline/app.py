from fastapi import FastAPI 
from pydantic import BaseModel
import json
import requests
import yaml
from producer import create_producer
from contextlib import asynccontextmanager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "configs/config.yml") -> dict:
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    producer = create_producer(
        bootstrap_servers=pipeline_config["kafka"]["bootstrap_servers"],
        batch_size=pipeline_config["kafka"]["batch_size"],
        linger_ms=pipeline_config["kafka"]["linger_ms"],
        retries=pipeline_config["kafka"]["retries"],
        retry_interval=pipeline_config["kafka"]["retry_interval"],
    )

    app.state.producer = producer
    yield
    logger.info("Shutting down...")
    if hasattr(app.state, 'producer'):
        app.state.producer.flush()




app = FastAPI(lifespan=lifespan)



@app.post("/pipeline")
def pipeline(req: DetectionRequest): #-> list[RecognitionResponseV2]:
    detection_response = requests.post(f'{pipeline_config["detection_server"]}', json=req.model_dump()) 
    detection_result = detection_response.json()

    
    # 이미지 저장 없이, 탐지결과 이미지를 메세지 큐로 전달
    # 인식모델이 해당 topic을 subscribe하고 있음
    app.state.producer.produce(
        topic=pipeline_config["kafka"]["topic"],
        value=json.dumps(detection_result).encode('utf-8')
    )
    return {'status': 'processing', 'file_name': req.file_name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)




