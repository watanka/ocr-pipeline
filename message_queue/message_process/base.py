from abc import ABC, abstractmethod
from typing import List, Dict, Any

class MessageProcessStrategy(ABC):
    """메시지 처리 전략 인터페이스"""
    
    @abstractmethod
    async def should_process(self, messages: List[Dict[str, Any]]) -> bool:
        """메시지를 처리해야 하는지 여부를 결정"""
        raise NotImplementedError