from redis.asyncio import Redis
from typing import Any, Dict, Optional
import logging
from ..base import MessageQueue
import asyncio

logger = logging.getLogger(__name__)

class RedisQueue(MessageQueue):
    def __init__(self, url: str):
        self.url = url
        self.redis: Optional[Redis] = None
    
    async def connect(self) -> None:
        """Redis 서버에 연결"""
        try:
            self.redis = Redis.from_url(self.url, decode_responses=True)
            # 연결 테스트
            await self.redis.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
    
    async def disconnect(self) -> None:
        """Redis 서버와 연결 해제"""
        if self.redis:
            await self.redis.close()
        logger.info("Disconnected from Redis")
    
    async def publish(self, queue_name: str, message: Dict[str, Any]) -> None:
        """메시지를 Redis 큐에 발행"""
        if not self.redis:
            await self.connect()
        
        try:
            message_body = self._serialize_message(message)
            await self.redis.rpush(queue_name, message_body)
            logger.info(f"Published message to queue: {queue_name}")
        except Exception as e:
            logger.error(f"Failed to publish message: {str(e)}")
            raise
    
    async def subscribe(self, queue_name: str, callback) -> None:
        """Redis 큐에서 메시지를 구독"""
        if not self.redis:
            await self.connect()
        
        try:
            while True:
                # 블로킹 방식으로 메시지 가져오기 (타임아웃 1초)
                try:
                    result = await self.redis.blpop(queue_name, timeout=1)
                    if result:
                        _, message_body = result
                        await self._handle_message(message_body, callback)
                except TimeoutError:
                    # 타임아웃은 정상적인 상황
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    continue
                
                await asyncio.sleep(0.1)  # CPU 사용량 감소
        except asyncio.CancelledError:
            logger.info("Subscription cancelled")
            raise
        except Exception as e:
            logger.error(f"Failed to subscribe to queue: {str(e)}")
            raise 