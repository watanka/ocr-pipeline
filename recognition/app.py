from fastapi import FastAPI
from pydantic import BaseModel
import logging
import yaml
from pathlib import Path


app = FastAPI()


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
    config = load_config()
    recognition_config = config["recognition"]


    logging.info(f'전처리: {recognition_config["preprocess"]}')
    logging.info(f'{recognition_config["model"]}, {recognition_config["model_path"]} 동작 검증...')
    logging.info(f'모델 추론 시작...')
    logging.info(f'모델 추론 완료!')
    logging.info(f'후처리: {recognition_config["postprocess"]}')


    return [
        RecognitionResponseV2(
            id=req.id,
            file_name=req.file_name,
            text=f'hello',
            confidence=0.90
        )
        for req in crop_requests
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

