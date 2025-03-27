from typing import Dict, Type
from .base import MessageQueue
from .rabbitmq.queue import RabbitMQ
from .redis.queue import RedisQueue

class MessageQueueFactory:
    """메시지 큐 구현체 생성 팩토리"""
    
    _implementations: Dict[str, Type[MessageQueue]] = {
        "rabbitmq": RabbitMQ,
        "redis": RedisQueue
    }
    
    @classmethod
    def create(cls, queue_type: str, **kwargs) -> MessageQueue:
        """
        메시지 큐 구현체 생성
        
        Args:
            queue_type: 메시지 큐 타입 ("rabbitmq" 또는 "redis")
            **kwargs: 구현체 생성에 필요한 추가 인자
            
        Returns:
            MessageQueue: 생성된 메시지 큐 구현체
            
        Raises:
            ValueError: 지원하지 않는 메시지 큐 타입인 경우
        """
        if queue_type not in cls._implementations:
            raise ValueError(f"Unsupported queue type: {queue_type}")
        
        return cls._implementations[queue_type](**kwargs) 