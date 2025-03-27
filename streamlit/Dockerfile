FROM python:3.9-slim

WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# Streamlit 포트 설정
EXPOSE 8501

# Streamlit 실행
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"] 