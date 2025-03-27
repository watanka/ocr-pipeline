import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List
from .base import MessageProcessStrategy
from datetime import datetime
from typing import List, Dict, Any
from common.logger import setup_logger, LOG_FORMATS

logger = setup_logger(
    'hybrid',
    format_string=LOG_FORMATS['DETAILED']
)

class HybridStrategy(MessageProcessStrategy):
    """배치 사이즈와 시간 기반 처리 전략"""
    
    def __init__(self, max_batch_size: int, wait_time: float):
        self.max_batch_size = max_batch_size
        self.wait_time = wait_time
        self.message_buffer = []
        self.last_process_time = datetime.now()
        self.lock = asyncio.Lock()  # 동시 접근 방지
        self.process_event = asyncio.Event()  # 강제 트리거 이벤트

        # 주기적으로 대기 시간을 체크하는 백그라운드 작업 실행
        asyncio.create_task(self.batch_timer())

    async def batch_timer(self):
        """주기적으로 배치 처리 확인"""
        logger.info("배치 타이머 시작")
        while True:
            logger.info(f"배치 타이머 대기 중... (대기 시간: {self.wait_time}초)")
            await asyncio.sleep(self.wait_time)
            async with self.lock:
                time_elapsed = (datetime.now() - self.last_process_time).total_seconds()
                logger.info(f"현재 버퍼 크기: {len(self.message_buffer)}개, 경과 시간: {time_elapsed:.2f}초")
                if self.message_buffer and time_elapsed >= self.wait_time and not self.process_event.is_set():
                    logger.info("최대 대기 시간 초과, 메시지 처리 시작")
                    self.process_event.set()  # 이벤트 트리거
                else:
                    logger.info("처리 조건 미충족, 계속 대기")

    async def should_process(self, messages: List[Dict[str, Any]]) -> bool:
        """메시지를 추가하고, 배치 처리할지 여부를 결정"""
        async with self.lock:
            self.message_buffer.extend(messages)
            total_regions = sum(len(msg.get('regions', [])) for msg in self.message_buffer)
            time_elapsed = (datetime.now() - self.last_process_time).total_seconds()

            should_process = total_regions >= self.max_batch_size or time_elapsed >= self.wait_time

            logger.info(f"메시지 버퍼: {len(self.message_buffer)}개, 대기 시간: {time_elapsed:.2f}s")

            if should_process:
                self.process_event.set()  # 이벤트 트리거
                self.last_process_time = datetime.now()

            return should_process
    
    async def get_messages(self) -> List[Dict[str, Any]]:
        """버퍼의 메시지를 반환하고 비움"""
        async with self.lock:
            messages = self.message_buffer.copy()
            self.message_buffer.clear()
            return messages
        
