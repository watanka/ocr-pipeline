import os
import cv2
import numpy as np

from processors import base64_to_image
from common.schema.detection import DetectionResponse
from common.schema.recognition import BatchRecognitionResponse

# SingleDetection id: {id}_{file_name}_{idx}

# 저장항목
# 폴더 이름: {request_id}
# crop 폴더: SingleDetection 이미지들
# 좌표, 텍스트들 -> result.txt
# original image 이름: {filename}

def split_id(identifier: str):
    # 오른쪽에서부터 두 번만 분리
    request_id, rest = identifier.split('_', 1)
    filename, idx = rest.rsplit('_', 1)
    
    return request_id, filename, idx


def save_result(original_image: np.array, 
                filename: str,
                request_id_full: str,
                detection_result: DetectionResponse, 
                recognition_result: BatchRecognitionResponse,
                save_dir: str = "results/"
                ):

    request_id, _ = request_id_full.rsplit('_', 1)
    # 폴더 생성
    os.makedirs(os.path.join(save_dir, request_id), exist_ok=True)
    os.makedirs(os.path.join(save_dir, request_id, "crop"), exist_ok=True)
    # original image 저장
    cv2.imwrite(os.path.join(save_dir, request_id, filename), original_image)

    # crop 이미지 저장
    for region in detection_result.regions:
        _, filename, idx = split_id(region.request_id)
        crop_image = base64_to_image(region.image)
        if crop_image is not None:
            cv2.imwrite(os.path.join(save_dir, request_id, "crop", f"{idx}.jpg"), crop_image)

    # result.txt 저장
    with open(os.path.join(save_dir, request_id, "result.txt"), "w") as f:
        for result in recognition_result.results:
            line = "\t".join(str(b) for b in result.bbox) + "\t" + result.text + "\n"
            f.write(line)

    
