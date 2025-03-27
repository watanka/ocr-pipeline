import os
import time
import requests

# 테스트 이미지 경로
TEST_IMAGE_DIR = 'test_image'
# 서버 URL
OCR_URL = 'http://127.0.0.1:8002/ocr'
BATCH_OCR_URL = 'http://127.0.0.1:8002/batch_ocr'

# 문서 수 N
N = 100  # 원하는 문서 수로 설정

def load_test(url, image_path, n):
    total_time = 0
    for i in range(n):
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'multipart/form-data')}
            start_time = time.time()
            response = requests.post(url, files=files)
            end_time = time.time()
            total_time += (end_time - start_time)
            if response.status_code != 200:
                print(f"Error: {response.status_code}, {response.text}")
            else:
                print(f"Request {i+1}/{n} completed in {end_time - start_time:.2f} seconds")
    return total_time

def main():
    # 테스트 이미지 목록
    image_files = [os.path.join(TEST_IMAGE_DIR, f) for f in os.listdir(TEST_IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # /ocr 테스트
    # print("Starting /ocr load test...")
    # ocr_total_time = 0
    # for image_path in image_files:
    #     ocr_total_time += load_test(OCR_URL, image_path, N // len(image_files))
    # print(f"Total time for /ocr: {ocr_total_time:.2f} seconds")

    # /batch_ocr 테스트
    print("Starting /batch_ocr load test...")
    batch_ocr_total_time = 0
    for image_path in image_files:
        batch_ocr_total_time += load_test(BATCH_OCR_URL, image_path, N // len(image_files))
    print(f"Total time for /batch_ocr: {batch_ocr_total_time:.2f} seconds")

if __name__ == "__main__":
    main()
