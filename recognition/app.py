from fastapi import FastAPI
from model import RecognitionRequestV2, RecognitionResponseV2
from recognition_model import process_recognition_model
from consumer import OCRConsumer
from contextlib import asynccontextmanager
import yaml


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup

    config = load_config()
    recognition_config = config["recognition"]
    app.state.config = recognition_config
    consumer = OCRConsumer(bootstrap_servers=config['pipeline']['kafka']['bootstrap_servers'],
                            topic=config['pipeline']['kafka']['topic'],
                            group_id=config['pipeline']['kafka']['group_id'],
                            )
    consumer.run()
    yield
    # Shutdown
    consumer.stop()  # consumer에 stop 메서드 추가 필요

app = FastAPI(lifespan=lifespan)



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


@app.post("/recognition")
def recognition(crop_requests: list[RecognitionRequestV2]) -> list[RecognitionResponseV2]:
    return process_recognition_model(app.state.config, crop_requests)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

