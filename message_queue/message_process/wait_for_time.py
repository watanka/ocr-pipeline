from .base import MessageProcessStrategy
from datetime import datetime
from typing import List, Dict, Any

class TimeBasedStrategy(MessageProcessStrategy):
    """일정 시간이 지나면 메시지 처리"""
    
    def __init__(self, wait_time: float):
        self.wait_time = wait_time
        self.last_process_time = datetime.now()
        self.message_buffer = []
    
    async def should_process(self, messages: List[Dict[str, Any]]) -> bool:
        # 메시지 버퍼에 추가
        self.message_buffer.extend(messages)
        
        # 처리 조건 확인
        time_elapsed = (datetime.now() - self.last_process_time).total_seconds()
        should_process = time_elapsed >= self.wait_time
        
        if should_process:
            self.last_process_time = datetime.now()
        
        return should_process
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """현재 버퍼의 메시지를 반환하고 버퍼를 비움"""
        messages = self.message_buffer.copy()
        self.message_buffer = []
        return messages