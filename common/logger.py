import logging
import sys
from typing import Optional

def setup_logger(
    name: str,
    level: int = logging.WARNING,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    로거를 설정하고 반환합니다.
    
    Args:
        name: 로거 이름
        level: 로깅 레벨 (기본값: logging.INFO)
        format_string: 로그 포맷 문자열 (기본값: '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file: 로그 파일 경로 (기본값: None, 콘솔에만 출력)
    
    Returns:
        logging.Logger: 설정된 로거 객체
    """
    # 기본 포맷 문자열 설정
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 이미 핸들러가 있다면 추가하지 않음
    if logger.handlers:
        return logger
    
    # 포맷터 생성
    formatter = logging.Formatter(format_string)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 설정 (log_file이 지정된 경우)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 로깅 레벨 상수
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# 로그 포맷 상수
LOG_FORMATS = {
    'DEFAULT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'SIMPLE': '%(levelname)s - %(message)s',
    'DETAILED': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
}

# 사용 예시
if __name__ == '__main__':
    # 기본 로거 설정
    logger = setup_logger('example')
    logger.info('This is an info message')
    logger.error('This is an error message')
    
    # 파일 로깅이 포함된 로거 설정
    file_logger = setup_logger(
        'file_example',
        level=logging.DEBUG,
        format_string=LOG_FORMATS['DETAILED'],
        log_file='example.log'
    )
    file_logger.debug('This is a debug message')
    file_logger.info('This is an info message') 