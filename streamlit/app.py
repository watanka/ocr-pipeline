import streamlit as st
import requests
import base64
from PIL import Image
import io
import json
import uuid

API_URL = "http://pipeline:8002"

# 페이지 설정
st.set_page_config(
    page_title="OCR 시스템",
    page_icon="📝",
    layout="wide"
)

# 스타일 설정
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .image-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 30px;
    }
    .text-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    
    # 로그 레벨 설정
    log_level = st.selectbox(
        "로그 레벨",
        ["INFO", "DEBUG", "WARNING", "ERROR"],
        index=0
    )
    
    # 처리 방식 설정
    processing_mode = st.radio(
        "처리 방식",
        ["Direct", "Message Queue"],
        index=0
    )
    
    st.info("설정은 이미지 인식 시 자동으로 적용됩니다.")

def upload_file():
    """파일 업로드 처리"""
    uploaded_file = st.file_uploader("이미지 파일을 선택하세요", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # 파일 정보 표시
        file_details = {
            "파일명": uploaded_file.name,
            "파일 크기": f"{uploaded_file.size / 1024:.2f} KB",
            "파일 타입": uploaded_file.type
        }
        st.write("### 파일 정보")
        for key, value in file_details.items():
            st.write(f"{key}: {value}")
        
        # 이미지 미리보기
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", use_column_width=True)
        
        return uploaded_file
    return None

def process_ocr_pipeline(uploaded_file, processing_mode):
    """OCR 파이프라인 실행"""
    try:
        # 파일을 바이트로 변환
        file_bytes = uploaded_file.getvalue()
        
        if processing_mode == "Direct":
            # API 방식: /ocr 엔드포인트 호출
            files = {
                'file': (uploaded_file.name, file_bytes, uploaded_file.type)
            }
            response = requests.post(f"{API_URL}/ocr", files=files)
        else:
            # Message Queue 방식: /batch_ocr 엔드포인트 호출
            files = {
                'file': (uploaded_file.name, file_bytes, uploaded_file.type)
            }
            response = requests.post(f"{API_URL}/batch_ocr", files=files)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            st.error(f"OCR 처리 중 오류 발생: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"OCR 처리 중 오류 발생: {str(e)}")
        return None

def main():
    st.title("OCR 서비스")
    
    # 파일 업로드 처리
    uploaded_file = upload_file()
    
    if uploaded_file is not None:
        if st.button("OCR 처리 시작"):
            with st.spinner("OCR 처리 중..."):
                # OCR 파이프라인 실행
                result = process_ocr_pipeline(uploaded_file, processing_mode)
                
                if result:
                    # 결과 표시
                    st.success("OCR 처리 완료!")
                    st.write("### 처리 결과")
                    
                    # 결과 이미지 표시
                    if "result_image" in result:
                        result_image = Image.open(io.BytesIO(base64.b64decode(result["result_image"])))
                        st.image(result_image, caption="인식 결과 이미지", use_column_width=True)
                    
                    # 텍스트 결과 표시
                    if "regions" in result and result["regions"]:
                        st.subheader("인식된 텍스트 영역")
                        for i, region in enumerate(result["regions"], 1):
                            with st.expander(f"영역 #{i} (ID: {region.get('id', i)})"):
                                st.markdown(f"**텍스트:** {region.get('text', '텍스트를 인식할 수 없음')}")
                                if 'box' in region:
                                    box = region['box']
                                    st.markdown(f"""
                                        **위치:** ({box[0][0]}, {box[0][1]}) - ({box[2][0]}, {box[2][1]})
                                    """)
                    else:
                        st.info("텍스트 영역을 찾을 수 없습니다.")
                else:
                    st.error("OCR 처리 실패")

if __name__ == "__main__":
    main() 