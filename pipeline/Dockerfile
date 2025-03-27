FROM python:3.8-slim

WORKDIR /app

# 기본 패키지 설치 (OpenCV 의존성 추가)
RUN apt-get update && apt-get install -y \
    wget \
    libgl1-mesa-glx \
    ffmpeg \
    libfontconfig1 \
    libxcb1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*



# 서비스 코드 복사 및 설치
COPY pipeline /app/
RUN pip install -r requirements.txt

# common 패키지 설치
COPY common /app/common
RUN pip install -e /app/common

# 포트 개방
EXPOSE 8002

# 볼륨 설정
VOLUME /app/static/uploads

# 서버 실행
CMD ["python", "app.py"] 