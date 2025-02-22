# docker image build
docker build -t detection -f Dockerfile.detection .
docker build -t recognition -f Dockerfile.recognition .
docker build -t pipeline -f Dockerfile.pipeline .

# detection
docker run --rm -d -p8000:8000 -v $(pwd)/../configs:/app/configs detection

# recognition
docker run --rm -d -p8001:8001 -v $(pwd)/../configs:/app/configs recognition

# pipeline
docker run --rm -p8002:8002 pipeline