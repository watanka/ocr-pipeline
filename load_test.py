import os, shutil, glob
import time
import asyncio
import aiohttp
import httpx
# 테스트 이미지 경로
TEST_IMAGE_DIR = 'test_image'
RESULT_DIR = 'pipeline/results'
# 서버 URL
OCR_URL = 'http://127.0.0.1:8002/ocr'
BATCH_OCR_URL = 'http://127.0.0.1:8002/batch_ocr'

# 문서 수 N
N = 500  # 원하는 문서 수로 설정

def count_files():
    """저장된 파일 개수 확인. result.txt가 있는 경우 완료 판단"""
    folder_path = os.path.join(RESULT_DIR)
    print(f"Counting files in {folder_path}...")
    print('len(os.listdir(folder_path))', len(os.listdir(folder_path)))
    print('os.path.exists(folder_path)', os.path.exists(folder_path))
    print('glob.glob(os.path.join(folder_path, "*/result.txt"))', len(glob.glob(os.path.join(folder_path, "*/result.txt"))))
    return min(len(os.listdir(folder_path)), len(glob.glob(os.path.join(folder_path, '*/result.txt'))))

def wait_for_completion():
    """모든 파일이 저장될 때까지 대기"""
    start = time.time()
    print("Waiting for all files to be saved...")
    while count_files() < N:
        time.sleep(0.1)  # 100ms마다 체크
    end = time.time()
    return end - start

def clear_result():
    """결과 파일 삭제"""
    folder_path = os.path.join(RESULT_DIR)
    print(f"Clearing results in {folder_path}...")
    shutil.rmtree(folder_path)

def load_test(session, url, image_path):
    """동기 이미지 파일 전송"""
    print(f"Sending request for {os.path.basename(image_path)} to {url}")
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'multipart/form-data')}
        
        response = session.post(url, files=files)
        print('요청 전송')
        status = response.status_code
        if status == 200:
            print(f"Request for {os.path.basename(image_path)} completed successfully!")
        else:
            print(f"Request for {os.path.basename(image_path)} failed with status {status}")
        return status

def run_load_test(url, image_files, n, interval=0):
    with httpx.Client(timeout=60) as client:
        for image_path in image_files:
            for _ in range(max(n // len(image_files), 1)):
                print(f"Sending request for {os.path.basename(image_path)}")  # 디버깅 메시지 추가
                load_test(client, url, image_path)
                time.sleep(interval)  # 요청 간격 조정 (0.1초)

def main():
    # 테스트 이미지 목록
    image_files = [os.path.join(TEST_IMAGE_DIR, f) for f in os.listdir(TEST_IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} image files for testing.")
    os.makedirs(RESULT_DIR, exist_ok=True)
    clear_result()

    # /ocr 테스트
    print("Starting /ocr load test...")
    start_time = time.time()
    run_load_test(OCR_URL, image_files, N)
    end_time = time.time()
    ocr_total_time = end_time - start_time
    ocr_total_time += wait_for_completion()
    print(f"Total time for /ocr: {ocr_total_time:.2f} seconds")


    time.sleep(30)
    clear_result()

    # 완료 후 결과 확인

    # /batch_ocr 테스트
    print("Starting /batch_ocr load test...")
    start_time = time.time()
    run_load_test(BATCH_OCR_URL, image_files, N)
    end_time = time.time()
    batch_ocr_total_time = end_time - start_time
    batch_ocr_total_time += wait_for_completion()
    print(f"Total time for /batch_ocr: {batch_ocr_total_time:.2f} seconds")
    
    
    

if __name__ == "__main__":
    main()
