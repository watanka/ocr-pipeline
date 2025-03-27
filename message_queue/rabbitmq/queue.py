import aio_pika
from typing import Any, Dict, Optional, Callable
import logging
from ..base import MessageQueue
import asyncio
import time
import json
from common.schema import BatchRecognitionResponse, RecognitionResult
import traceback
from common.logger import setup_logger, LOG_FORMATS
from ..message_process.base import MessageProcessStrategy
from ..bucket import BatchBucket
# 로거 설정
logger = setup_logger(
    'message_queue',
    format_string=LOG_FORMATS['DETAILED']
)

class RabbitMQ(MessageQueue):
    def __init__(self, url: str):
        self.url = url
        self.connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.queues: Dict[str, aio_pika.Queue] = {}  # 선언된 큐 저장
    
    async def connect(self) -> None:
        """RabbitMQ 서버에 연결"""
        try:
            # 연결 설정
            self.connection = await aio_pika.connect(self.url)
            self.channel = await self.connection.channel()
            logger.info("Connected to RabbitMQ")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            raise
    
    async def declare_queue(self, queue_name: str) -> None:
        """큐 선언 및 익스체인지와 바인딩"""
        if queue_name not in self.queues:
            queue = await self.channel.declare_queue(
                queue_name,
                durable=True  # durable 옵션 추가
            )
            self.queues[queue_name] = queue
            logger.info(f"Declared and bound queue: {queue_name}")
    
    async def publish(self, queue_name: str, message: Dict[str, Any]) -> None:
        """메시지를 RabbitMQ 큐에 발행"""
        if not self.channel:
            await self.connect()
        
        try:
            # 큐가 아직 선언되지 않았다면 선언
            if queue_name not in self.queues:
                await self.declare_queue(queue_name)
            
            # 딕셔너리를 JSON 문자열로 변환 후 인코딩
            message_body = json.dumps(message).encode()

            await self.channel.default_exchange.publish(
                aio_pika.Message(
                    body=message_body,
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key=queue_name
            )
            logger.info(f"Published message to queue: {queue_name}")
        except Exception as e:
            logger.error(f"Failed to publish message: {str(e)}")
            raise
    
    async def subscribe(self, queue_name: str, callback) -> None:
        if not self.channel:
            await self.connect()

        if queue_name not in self.queues:
            await self.declare_queue(queue_name)
        
        async def process_message(message: aio_pika.IncomingMessage):
            async with message.process():
                try:
                    # 바이트 -> 문자열 -> JSON 딕셔너리
                    message_body = message.body.decode()
                    body = json.loads(message_body)
                    
                    # callback 실행
                    await callback(body)
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
        
        try:
            # 메시지 구독 시작
            await self.queues[queue_name].consume(process_message)
            logger.info(f"Started consuming messages from queue: {queue_name}")
            
            # 무한 대기
            while True:
                await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Failed to subscribe to queue: {str(e)}")
            raise
    
    async def disconnect(self) -> None:
        """RabbitMQ 서버와 연결 해제"""
        if self.connection:
            await self.connection.close()
            self.connection = None
            self.channel = None
            self.exchange = None
            self.queues.clear()
            logger.info("Disconnected from RabbitMQ") 
            
    async def consume(
        self, 
        queue_name: str, 
        timeout: float = 60.0, 
        filter_fn: Callable = None
    ) -> BatchRecognitionResponse:
        start_time = time.time()
        result = None
        future = asyncio.Future()
        
        async def process_str_message(message: aio_pika.IncomingMessage):
            nonlocal result
            async with message.process():
                try:
                    message_body = message.body.decode()
                    body = json.loads(message_body)
                    logger.info(f"Received message: {body.keys()}")
                    
                    if filter_fn is None or filter_fn(body):

                        recognition_results = body.get("results", [])
                        recognition_results = [RecognitionResult.model_validate(result) for result in recognition_results]
                        result = BatchRecognitionResponse(request_id=body.get("request_id"), results=recognition_results)
                        future.set_result(result)
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    logger.error(traceback.format_exc())
        
        try:
            # 큐가 아직 선언되지 않았다면 선언
            if queue_name not in self.queues:
                await self.declare_queue(queue_name)
            
            # 메시지 구독 시작
            await self.queues[queue_name].consume(process_str_message)
            logger.info(f"Started consuming messages from queue: {queue_name}")
            
            try:
                # 타임아웃까지 대기
                result = await asyncio.wait_for(future, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for message from queue: {queue_name}")
                raise TimeoutError(f"Queue {queue_name}에서 메시지 수신 타임아웃")
            
        except Exception as e:
            logger.error(f"Failed to consume message: {str(e)}")
            raise


    async def handle_str_request(self, queue_name: str, callback, strategy: MessageProcessStrategy):
        """메시지를 처리할지 여부를 결정하는 로직"""
        if not self.channel:
            await self.connect()

        if queue_name not in self.queues:
            await self.declare_queue(queue_name)
        
        async def process_message(message: aio_pika.IncomingMessage):
            async with message.process():
                try:
                    body = json.loads(message.body.decode())
                    await strategy.should_process([body])
                except Exception as e:
                    logger.error(f"메시지 처리 중 오류: {str(e)}")

        # 큐 소비 시작
        await self.queues[queue_name].consume(process_message)
        logger.info(f"{queue_name} 큐에서 메시지 수신 시작")

        # 배치 이벤트 감지 후 처리
        await self.batch_processor(callback, strategy)

    async def batch_processor(self, callback, strategy: MessageProcessStrategy):
        """배치가 준비되면 메시지를 처리"""
        while True:
            await strategy.process_event.wait()  # 이벤트 발생 시 실행
            strategy.process_event.clear()  # 이벤트 초기화
            messages = await strategy.get_messages()
            if messages:
                logger.info(f"배치 크기 {len(messages)}개 처리 시작")
                await callback(messages)  # 배치 처리 콜백 실행
        
    async def pour(self, bucket: BatchBucket, queue_name: str):
        await bucket.fill(self, queue_name)