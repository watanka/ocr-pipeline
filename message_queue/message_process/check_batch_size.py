from .base import MessageProcessStrategy
from typing import List, Dict, Any

class BatchSizeStrategy(MessageProcessStrategy):
    """배치 사이즈가 차면 메세지 처리"""
    
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.message_buffer = []

    async def should_process(self, messages: List[Dict[str, Any]]) -> bool:
        self.message_buffer.extend(messages)
        total_regions = sum(len(msg['regions']) for msg in self.message_buffer)
        should_process = total_regions >= self.batch_size
        return should_process

    def get_messages(self) -> List[Dict[str, Any]]:
        """현재 버퍼의 메시지를 반환하고 버퍼를 비움"""
        messages = self.message_buffer.copy()
        self.message_buffer = []
        return messages