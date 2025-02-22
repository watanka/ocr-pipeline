from fastapi import FastAPI
from pydantic import BaseModel
import logging
import os, yaml


app = FastAPI()

class DetectionRequest(BaseModel):
    file_name: str
    image: str # base64 encoded

class DetectionResponseV1(BaseModel):
    file_name: str
    bbox_type: str
    bbox: list[list[int]]
    crop_file_name: str

class DetectionResponseV2(BaseModel):
    id: str
    file_name: str
    image: str # base64 encoded


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


@app.post("/detection")
def detection(req: DetectionRequest) -> list[DetectionResponseV1 | DetectionResponseV2]:
    config = load_config()
    detection_config = config["detection"]


    logging.info(f'전처리: {detection_config["preprocess"]}')
    logging.info(f'{detection_config["model"]}, {detection_config["model_path"]} 동작 검증...')
    logging.info(f'모델 추론 시작...')
    logging.info(f'모델 추론 완료!')
    logging.info(f'후처리: {detection_config["postprocess"]}')


    return [
        DetectionResponseV2(
            id=f'{os.path.basename(os.path.splitext(req.file_name)[0])}_1', # 파일이름에 대한 전처리
            file_name=req.file_name,
            image="random_crop_image1"
        ),
        DetectionResponseV2(
            id=f'{os.path.basename(os.path.splitext(req.file_name)[0])}_2', # 파일이름에 대한 전처리
            file_name=req.file_name,
            image="random_crop_image2"
        )
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

