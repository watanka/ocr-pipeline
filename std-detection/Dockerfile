FROM python:3.8-slim

WORKDIR /app

# 기본 패키지 설치 (OpenCV 의존성 추가)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    ffmpeg \
    libfontconfig1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# 필요한 패키지 설치 (Flask와 Werkzeug 버전 지정)

COPY ./std-detection /app/
RUN pip install -r requirements.txt

COPY common /app/common
RUN pip install -e /app/common

# 포트 개방
EXPOSE 8000

# 서버 실행
CMD ["python", "api.py"] 