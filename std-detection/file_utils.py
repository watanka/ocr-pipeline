"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
import base64

def byte2image(image_bytes: bytes) -> np.ndarray:
    # Base64 이미지 디코딩
    img_bytes = base64.b64decode(image_bytes)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 이미지가 없는 경우
    if img is None:
        raise ValueError("Failed to decode image")
        
    # BGR에서 RGB로 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 이미지 정보 추출
    height, width, channels = img.shape
    return img, height, width

def crop_text_region(image, xmin, ymin, xmax, ymax):
    return image[ymin:ymax, xmin:xmax]


def image2base64(image) -> str:
    # 이미지를 Base64로 인코딩
    if image.size > 0:
        # BGR로 변환 (CV2 저장을 위해)
        region_img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode('.jpg', region_img_bgr)
        if success:
            region_img_base64 = base64.b64encode(buffer).decode('utf-8')
        else:
            region_img_base64 = ""
    else:
        region_img_base64 = ""
    return region_img_base64

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    # 텍스트 영역 추출
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    _, text_score = cv2.threshold(textmap, low_text, 1, 0)
    _, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
    
    # 텍스트 점수와 링크 점수 결합
    text_score_comb = np.clip(text_score + link_score, 0, 1)

    # 연결 요소 감지
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)
    
    det = []
    mapper = []

    # 각 연결 요소 처리
    for k in range(1, nLabels):
        # 크기, 높이, 영역 계산
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # 연결 요소 마스크 생성
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0  # 링크 영역만 있는 부분 제거

        # 텍스트 점수 계산
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # 경계 검사
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # 다각형 추출
        if poly:
            # 외곽선 찾기
            contours, _ = cv2.findContours(segmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            
            # 가장 큰 외곽선 선택
            contour = contours[0]
            for cont in contours:
                if cv2.contourArea(cont) > cv2.contourArea(contour):
                    contour = cont
            
            # 다각형 근사화
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 단순화된 다각형
            points = approx.reshape((-1, 2))
            
            # 너무 많은 점 있으면 단순화
            if len(points) > 20:
                contour = contour.reshape((-1, 2))
                l = len(contour)
                step = l // 20
                points = contour[step-1::step, :]
                points = np.append(points, [points[0]], axis=0)
            
            # 마지막 결과 저장
            mapper.append(k)
            det.append(points)
        else:
            # bounding box 생성
            contours, _ = cv2.findContours(segmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            
            # 외곽선 기반 경계 상자 생성
            contour = contours[0]
            for cont in contours:
                if cv2.contourArea(cont) > cv2.contourArea(contour):
                    contour = cont
            
            # 회전된 경계 상자 계산
            points = cv2.boxPoints(cv2.minAreaRect(contour))
            
            # 마지막 결과 저장
            mapper.append(k)
            det.append(points)

    return det, labels

def adjustResultCoordinates(polys, ratio_w, ratio_h):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w, ratio_h)
    return polys 