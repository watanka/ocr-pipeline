from datetime import datetime, timedelta
import asyncio
import aio_pika
import json
from common.logger import setup_logger, LOG_FORMATS
from .base import MessageQueue

logger = setup_logger(
    'bucket',
    format_string=LOG_FORMATS['DETAILED']
)

class BatchBucket:
    def __init__(self, max_batch_size: int, wait_time: float):
        self.max_batch_size = max_batch_size
        self.wait_time = timedelta(seconds=wait_time)
        self.batch = []
        self.last_process_time = datetime.now()
        self.lock = asyncio.Lock()
    

    async def fill(self, message_queue: MessageQueue, queue_name: str):
        async def process_batch(message: aio_pika.IncomingMessage):
            async with self.lock:
                async with message.process():
                    try:
                        body = json.loads(message.body.decode())
                        self.batch.append(body)
                    except Exception as e:
                        logger.error(f"메시지 처리 중 오류: {str(e)}")
        if queue_name not in message_queue.queues:
            await message_queue.declare_queue(queue_name)
        
        await message_queue.queues[queue_name].consume(process_batch)
        logger.info(f"{queue_name} 큐에서 메시지 수신 시작")

    
    async def get_batch(self):
        async with self.lock:
            return self.batch.copy()

    def is_ready(self):
        return self.batch and \
            (len(self.batch) >= self.max_batch_size or \
             datetime.now() - self.last_process_time >= self.wait_time) 

