import logging

from model import RecognitionRequestV2, RecognitionResponseV2


def process_recognition_model(config, crop_requests: RecognitionRequestV2) -> list[RecognitionResponseV2]:
    logging.info(f'전처리: {config["preprocess"]}')
    logging.info(f'{config["model"]}, {config["model_path"]} 동작 검증...')
    logging.info(f'모델 추론 시작...')
    logging.info(f'모델 추론 완료!')
    logging.info(f'후처리: {config["postprocess"]}')


    return [
        RecognitionResponseV2(
            id=req.id,
            file_name=req.file_name,
            text=f'hello',
            confidence=0.90
        )
        for req in crop_requests
    ]