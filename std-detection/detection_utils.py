import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import traceback
def copyStateDict(state_dict):
    """
    PyTorch 상태 딕셔너리를 복사합니다.
    """
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = {}
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def preprocess_image(image):
    """
    이미지를 전처리합니다.
    """
    # RGB 변환
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image

def draw_detection_boxes(image, bboxes):
    """
    감지된 텍스트 박스를 이미지에 그립니다.
    """
    result_image = image.copy()
    for i, box in enumerate(bboxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        cv2.polylines(result_image, [poly.reshape((-1, 1, 2))], True, (0, 0, 255), 2)
    
    return result_image

def crop_text_regions(image, bboxes):
    """
    텍스트 영역을 잘라냅니다.
    """
    cropped_regions = []
    positions = []
    
    for box in bboxes:
        poly = np.array(box).astype(np.int32)
        x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
        x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
        
        # 이미지에서 텍스트 영역 추출
        text_img = image[y_min:y_max, x_min:x_max]
        
        # 위치 정보 생성
        position = {
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max),
            "width": int(x_max - x_min),
            "height": int(y_max - y_min)
        }
        
        cropped_regions.append(text_img)
        positions.append(position)
    
    return cropped_regions, positions

def initialize_model(model, model_path):
    """
    CRAFT 모델을 초기화합니다.
    """
    logger = None
    try:
        import logging
        logger = logging.getLogger("std-detection")
        
        # CUDA 확인
        if torch.cuda.is_available():
            if logger:
                logger.info("CUDA is available, using GPU")
            model.load_state_dict(copyStateDict(torch.load(model_path)))
            model = model.cuda()
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = False
        else:
            if logger:
                logger.info("CUDA is not available, using CPU")
            model.load_state_dict(copyStateDict(torch.load(model_path, map_location='cpu')))
    except Exception as e:
        if logger:
            logger.error(f"Error initializing model: {str(e)}")
            logger.error(traceback.format_exc())
        raise
    
    # 모델을 평가 모드로 설정
    model.eval()
    return model 