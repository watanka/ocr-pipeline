import os
import cv2
import numpy as np
import base64
import logging
import traceback
from pathlib import Path
from common.logger import setup_logger, LOG_FORMATS

logger = setup_logger(
    'pipeline_utils',
    format_string=LOG_FORMATS['DETAILED']
)

def draw_text_boxes(image_base64, regions_with_text):
    """인식된 텍스트와 함께 텍스트 박스를 이미지에 그립니다."""
    try:
        # 베이스64 디코딩
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 각 영역에 텍스트 박스와 인식된 텍스트 그리기
        for region in regions_with_text:
            if hasattr(region, 'bbox') and hasattr(region, 'text') and region.bbox is not None:
                # 박스 그리기
                bbox = np.array(region.bbox).astype(np.int32)
                cv2.polylines(image, [bbox.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
                
                # 텍스트 위치 계산
                x_min, y_min = np.min(bbox[:, 0]), np.min(bbox[:, 1])
                
                # 텍스트 그리기 (흰색 배경으로)
                text = region.text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                # 텍스트 배경 박스
                cv2.rectangle(image, (x_min, y_min - text_height - 5), 
                             (x_min + text_width, y_min), (255, 255, 255), -1)
                
                # 텍스트 그리기
                cv2.putText(image, text, (x_min, y_min - 5), 
                           font, font_scale, (0, 0, 0), thickness)
        
        # 이미지를 베이스64로 인코딩
        _, buffer = cv2.imencode('.jpg', image)
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return result_image_base64
    except Exception as e:
        logger.error(f"Error drawing text boxes: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def save_result_image(image_base64, folder, filename=None):
    """
    베이스64 인코딩된 이미지를 저장하고 경로를 반환합니다.
    """
    try:
        if not image_base64:
            return None
            
        os.makedirs(folder, exist_ok=True)
        
        if filename is None:
            import uuid
            filename = f"{uuid.uuid4()}.jpg"
            
        file_path = os.path.join(folder, filename)
        
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(image_base64))
        
        # 웹에서 접근 가능한 URL 반환
        url_path = f"/static/{Path(folder).name}/{filename}"
        return url_path
    except Exception as e:
        logger.error(f"Error saving result image: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def encode_image_to_base64(image_path):
    """
    이미지 파일을 베이스64로 인코딩합니다.
    """
    try:
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
            encoded_img = base64.b64encode(img_bytes).decode('utf-8')
        return encoded_img
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        logger.error(traceback.format_exc())
        return None 
    

def draw_text_boxes_with_text(image, regions):
    """
    이미지에 텍스트 박스와 인식된 텍스트를 그립니다.
    
    Args:
        image: 원본 이미지
        regions: 텍스트 영역 정보 목록
        
    Returns:
        텍스트 박스와 인식된 텍스트가 표시된 이미지
    """
    # 이미지 복사
    result_img = image.copy()
    
    # 각 영역에 대해
    for region in regions:
        # 박스 정보 가져오기
        bbox = region.get("bbox", [])
        if not bbox:
            continue
        
        # 다각형 좌표를 NumPy 배열로 변환
        points = np.array(bbox).astype(np.int32)
        
        # 텍스트 정보 가져오기
        text = region.get("text", "")
        
        # 박스 그리기 (빨간색)
        cv2.polylines(result_img, [points], True, (0, 0, 255), 2)
        
        # 텍스트가 있는 경우 표시
        if text:
            # 텍스트 표시 위치 (박스 상단)
            text_pos = (points[0][0], points[0][1] - 10)
            
            # 텍스트 배경 (검은색 반투명)
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_w, text_h = text_size
            cv2.rectangle(result_img, 
                        (text_pos[0] - 5, text_pos[1] - text_h - 5), 
                        (text_pos[0] + text_w + 5, text_pos[1] + 5), 
                        (0, 0, 0), -1)
            
            # 텍스트 표시 (흰색)
            cv2.putText(result_img, text, text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result_img