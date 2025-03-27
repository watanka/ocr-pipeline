from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)

class MessageQueue(ABC):
    """메시지 큐 기본 인터페이스"""
    
    @abstractmethod
    async def connect(self) -> None:
        """메시지 큐 서버에 연결"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """메시지 큐 서버와 연결 해제"""
        pass
    
    @abstractmethod
    async def publish(self, queue_name: str, message: Dict[str, Any]) -> None:
        """메시지를 큐에 발행"""
        pass
    
    @abstractmethod
    async def subscribe(self, queue_name: str, callback) -> None:
        """큐에서 메시지를 구독"""
        pass
    
    def _serialize_message(self, message: Dict[str, Any]) -> str:
        """메시지를 JSON 문자열로 직렬화"""
        return json.dumps(message)
    
    def _deserialize_message(self, message: str) -> Dict[str, Any]:
        """JSON 문자열을 메시지로 역직렬화"""
        return json.loads(message)
    
    async def _handle_message(self, message: Any, callback) -> None:
        """메시지 처리 래퍼"""
        try:
            if isinstance(message, str):
                message = self._deserialize_message(message)
            await callback(message)
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            raise 