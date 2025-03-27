## OCR 파이프라인

OCR(Optical Character Recognition) 파이프라인은 이미지에서 텍스트 영역을 탐지하고, 탐지한 영역에 포함된 문자를 인식하는 시스템입니다.

## 시스템 아키텍처

이 시스템은 다음과 같은 세 가지 서비스로 구성되어 있습니다:

1. **STD Detection (Text Detection)**: 이미지에서 텍스트 영역을 감지합니다.
2. **STR Recognition (Text Recognition)**: 감지된 텍스트 영역에서 텍스트를 인식합니다.
3. **Pipeline**: 두 서비스를 연결하고 웹 인터페이스를 제공합니다.

## 실행 방법

### 사전 요구 사항
- Docker
- Docker Compose
- NVIDIA GPU (선택 사항)

### 실행하기

```bash
# 도커 이미지 빌드
make build

# 서비스 시작
make up

# 로그 확인
make logs
```

### 사용하기

웹 인터페이스를 통해 OCR 서비스에 접근할 수 있습니다.
http://localhost:8501

## 성능 모니터링 및 테스트

OCR 파이프라인은 성능 모니터링 및 테스트 기능을 제공합니다. 자세한 내용은 [monitoring/README.md](monitoring/README.md)를 참조하세요.


**부하 테스트**  

- 순차처리(/ocr): STD 결과로 나온 박스를 STR이 순차적으로 처리함
- 배치처리(/pipeline): STD 결과를 메세지 큐를 통해 전송하면, 최대 배치 사이즈가 차거나 대기 시간이 지나기 전까지 배치를 모은 후 STR 모델로 처리함



[파악해야하는 항목]
- STD 소요 시간: 평균 약 1.5초
- STR 소요 시간: 평균 약 1.8초 (박스 갯수에 따라 다름)
- 문서 당 평균 박스 수
- 최대 처리 배치 사이즈
   - 현재 gpu에서 처리할 수 있는 최대 배치사이즈는 (사용 가능 메모리) / (이미지 데이터). 
   - 사용 가능 메모리는 RTX2070 8GB(8192MiB)를 기준으로 STD와 STR 모델 사이즈(3525MiB)와 기본으로 차지하고 있는 메모리(1567MiB)를 제외하고, 3100MiB
   - 이미지 데이터 1장의 사이즈는 3x64x100x 4bytes(float32)=0.0732MB
   - 이론적으로는 3100 / 0.0732 = 42349
   - 하지만 실제로는 300 - 400 정도의 배치 사이즈면 8192MiB용량을 거의 채우는 걸 확인할 수 있었다. 최대 배치사이즈를 중간값인 350으로 설정.
- BucketMonitor 대기 시간


N에 따라 일괄 처리 대비 배치 처리로 절약되는 시간을 정리한 테이블입니다. 각 N에 대해 절약되는 시간과 절약 비율(%)을 계산했습니다.
| 문서 수 N | 일괄 처리 시간 (초) | 배치 처리 시간 (초) | 절약 시간 (초) | 절약 비율(%) |
|----------------|--------------------|--------------------|---------------|--------------|
| 100 | 330 | 428.57 | 428.57 | 56.1% |
| 1,000 | 3,300 | 4,285.7 | 4,285.7 | 56.1% |
| 10,000 | 33,000 | 42,857 | 42,857 | 56.1% |

계산 방법
- 일괄처리 = STD + STR  
- 배치처리 = STD + 메세지큐 전송시간 + STR(일괄처리)  

- 예를 들어 문서 N장 처리하고, 문서 당 평균 박스 수가 200개라고 한다면,  
   - 일괄처리 = (Tstd + Tstr) x N = (1.5 + 1.8) x 100
   - 배치처리 => Tstd x N + Tstr x ((200 x N) / 350) = 1.5 x N + 1.8 x ((200 x N) / 350)





## API Endpoint
- `/detection`
- `/recognition`
- `/ocr`
- `/batch_ocr`

## 기술 스택

- STD: CRAFT (Character-Region Awareness For Text detection)
- STR: Deep Text Recognition Benchmark
- Docker & Docker Compose
- Flask (Python 웹 프레임워크)
