
import logging
import cv2
import base64
import numpy as np
import torch
import traceback
from utils import CTCLabelConverter, AttnLabelConverter
from common.schema import SingleDetection
from common.logger import setup_logger, LOG_FORMATS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = setup_logger(
    'str_recognition',
    format_string=LOG_FORMATS['DETAILED']
)
# 모델 옵션 설정
opt = {
    'Transformation': 'TPS',
    'FeatureExtraction': 'ResNet',
    'SequenceModeling': 'BiLSTM',
    'Prediction': 'Attn',
    'num_fiducial': 20,
    'imgH': 32,
    'imgW': 100,
    'character': '0123456789abcdefghijklmnopqrstuvwxyz',
    'input_channel': 1,
    'output_channel': 512,
    'hidden_size': 256,
    'num_fiducial': 20,
    'batch_max_length': 25,
    'rgb': True
}

# 변환기 초기화
if 'CTC' in opt['Prediction']:
    converter = CTCLabelConverter(opt['character'])
else:
    converter = AttnLabelConverter(opt['character'])
opt['num_class'] = len(converter.character)


def preprocess_image(img, opt):
    """
    이미지 전처리: 크기 조정 및 패딩
    """
    # 이미지 유효성 확인
    if img is None:
        logger.error("Input image is None in preprocess_image")
        raise ValueError("Input image is None")
    
    logger.info(f"Input image shape: {img.shape}, dtype: {img.dtype}")
    
    # 이미지 크기 조정
    target_h, target_w = opt['imgH'], opt['imgW']
    
    # 그레이스케일 이미지로 변환 (채널이 1인 경우)
    try:
        if opt['input_channel'] == 1:
            if len(img.shape) == 3:  # RGB 이미지인 경우
                logger.info("Converting RGB to grayscale")
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif len(img.shape) == 2:  # 이미 그레이스케일인 경우
                logger.info("Image is already grayscale")
            else:
                logger.warning(f"Unexpected image shape: {img.shape}")
        
        logger.info(f"After color conversion - shape: {img.shape}, dtype: {img.dtype}")
    except Exception as e:
        logger.error(f"Error during color conversion: {str(e)}")
        if len(img.shape) != 2:
            raise
    
    # 이미지 차원 확인
    if len(img.shape) == 2:
        h, w = img.shape
        logger.info(f"Working with grayscale image, h={h}, w={w}")
    else:
        h, w, c = img.shape
        logger.info(f"Working with color image, h={h}, w={w}, c={c}")
    
    # 비율 유지하면서 크기 조정
    ratio = w / float(h)
    if ratio > target_w / target_h:
        # 너비가 더 넓은 경우
        new_w = target_w
        new_h = int(new_w / ratio)
    else:
        # 높이가 더 높은 경우
        new_h = target_h
        new_w = int(ratio * new_h)
    
    logger.info(f"Resizing to new_h={new_h}, new_w={new_w}")
    
    # 크기 조정
    try:
        img = cv2.resize(img, (new_w, new_h))
        logger.info(f"After resize - shape: {img.shape}, dtype: {img.dtype}")
    except Exception as e:
        logger.error(f"Error during resize: {str(e)}")
        raise
    
    # 패딩 추가
    try:
        if len(img.shape) == 2:
            # 그레이스케일 이미지인 경우
            logger.info("Adding padding to grayscale image")
            padded_img = np.ones((target_h, target_w), dtype=np.uint8) * 255  # 흰색 배경
            padded_img[0:new_h, 0:new_w] = img
        else:
            # 컬러 이미지인 경우
            logger.info("Adding padding to color image")
            padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255  # 흰색 배경
            padded_img[0:new_h, 0:new_w, :] = img
        
        logger.info(f"Final padded image - shape: {padded_img.shape}, dtype: {padded_img.dtype}")
    except Exception as e:
        logger.error(f"Error during padding: {str(e)}")
        raise

    # 정규화 및 텐서 변환
    padded_img = padded_img.astype(np.float32) / 127.5 - 1.0
    
    # 채널 차원 추가 (그레이스케일인 경우)
    if len(padded_img.shape) == 2:
        padded_img = padded_img[np.newaxis, :, :]  # [1, H, W]
    
    # 텐서로 변환
    padded_img = torch.from_numpy(padded_img)
    logger.info(f"Tensor shape: {padded_img.shape}")
    
    return padded_img

def prepare_batch_tensor(images):
    """
    이미지 배치를 모델 입력 텐서로 변환
    이미지 순서가 섞이지 않도록 강제해야함
    """
    try:
        # 이미지 리스트를 배치 텐서로 변환
        processed_tensors = []
        for img in images:
            # 이미지 전처리
            processed_img = preprocess_image(img, opt)
            processed_tensors.append(processed_img)
        
        # 배치 텐서 생성 (batch_size, channel, height, width)
        batch_tensor = torch.stack(processed_tensors)
        batch_tensor = batch_tensor.to(device)
        
        logger.info(f"Batch tensor shape: {batch_tensor.shape}")
        return batch_tensor
    except Exception as e:
        logger.error(f"Error preparing batch tensor: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def post_process_text(text):
    """
    인식된 텍스트 후처리
    
    Args:
        text: 인식된 원본 텍스트
        
    Returns:
        후처리된 텍스트
    """
    # 공백 제거
    processed_text = text.strip()
    
    # 특수문자 처리
    # 예: 특정 문자 치환, 필터링 등
    
    # 여기에 추가적인 후처리 로직 추가 가능:
    # - 언어 확인
    # - 문맥 기반 오류 수정
    # - 정규화 등
    
    return processed_text

# 간소화된 인식 함수
# def recognize_image(img):
#     """
#     이미지에서 텍스트를 인식합니다.
    
#     Args:
#         img: 입력 이미지 (RGB)
        
#     Returns:
#         인식된 텍스트
#     """
#     monitor.start_timer("recognize_single_image")
    
#     try:
#         monitor.start_timer("image_preprocessing")
#         # 이미지 형태 확인
#         logger.info(f"recognize_image input - shape: {img.shape}, dtype: {img.dtype}")
        
#         # 이미지가 그레이스케일인지 확인
#         is_grayscale = len(img.shape) == 2
#         if is_grayscale:
#             logger.info("Input image is grayscale, converting to RGB for preprocessing")
#             # 그레이스케일을 RGB로 변환
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
#         # 이미지 전처리
#         processed_img = preprocess_image(img)
#         preproc_time = monitor.stop_timer("image_preprocessing")
#         logger.info(f"Processed image shape: {processed_img.shape}")
        
#         # 텐서로 변환
#         tensor_img = processed_img.to(device)
#         logger.info(f"Tensor shape: {tensor_img.shape}")
        
#         monitor.start_timer("text_prediction")
#         # 모델 추론
#         with torch.no_grad():
#             preds = model(tensor_img)
#             logger.info(f"Prediction shape: {preds.shape}")
        
#         # 결과 디코딩
#         if 'CTC' in opt['Prediction']:
#             _, preds_index = preds.max(2)
#             preds_str = converter.decode(preds_index, torch.IntTensor([preds.size(1)]))
#         else:
#             preds_index = preds.argmax(dim=2)
#             preds_str = converter.decode(preds_index, torch.IntTensor([preds.size(1)]))
        
#         pred_time = monitor.stop_timer("text_prediction")
        
#         logger.info(f"Raw recognition result: '{preds_str[0]}'")
        
#         # 비어있는 결과 처리
#         if not preds_str[0] or preds_str[0].strip() == '':
#             logger.warning("Empty recognition result. Checking confidence...")
#             # 신뢰도 확인 (필요시)
#             return "[인식 실패]"
            
#         total_time = monitor.stop_timer("recognize_single_image")
#         logger.info(f"Recognition completed in {total_time:.4f}s (preproc: {preproc_time:.4f}s, pred: {pred_time:.4f}s)")
        
#         return preds_str[0]
#     except Exception as e:
#         logger.error(f"Error in recognize_image: {str(e)}")
#         logger.error(traceback.format_exc())
#         monitor.stop_timer("image_preprocessing", force=True)
#         monitor.stop_timer("text_prediction", force=True)
#         monitor.stop_timer("recognize_single_image", force=True)
#         return "[오류 발생]"


def convert2image(region: SingleDetection) -> np.ndarray:
    # 이미지 디코딩
    img_bytes = base64.b64decode(region.image)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        logger.warning(f"Failed to decode image for region {i} (ID: {region.request_id})")
        raise ValueError("Failed to decode image")
    
    # BGR에서 RGB로 변환
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img
