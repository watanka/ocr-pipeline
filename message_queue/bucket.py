from datetime import datetime, timedelta
import asyncio
import aio_pika

from typing import Callable

import json
from common.logger import setup_logger, LOG_FORMATS
from .base import MessageQueue
from common.schema import DetectionResponse
logger = setup_logger(
    'bucket',
    format_string=LOG_FORMATS['DETAILED']
)

class BatchBucket:
    def __init__(self, max_batch_size: int, wait_time: float, data_handler: Callable, name: str = "batch_bucket"):
        self.max_batch_size = max_batch_size
        self.wait_time = timedelta(seconds=wait_time)
        self.data_handler = data_handler
        self.batch = []
        self.last_process_time = datetime.now()
        self.lock = asyncio.Lock()
        self.name = name

    async def add(self, data):
        async with self.lock:
            self.batch.append(self.data_handler(data))


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





class STDBucket:
    def __init__(self, max_batch_size: int, wait_time: float):
        self.max_batch_size = max_batch_size
        self.wait_time = timedelta(seconds=wait_time)
        self.batch = []
        self.last_process_time = datetime.now()
        self.lock = asyncio.Lock()
        
    async def add(self, data):
        async with self.lock:
            self.batch.extend(data.regions)

    async def get_batch(self):
        async with self.lock:
            batch_to_process = self.batch[:self.max_batch_size]
            self.batch = self.batch[self.max_batch_size:]
            return batch_to_process

    def is_ready(self):
        return self.batch and \
            (len(self.batch) >= self.max_batch_size or \
             datetime.now() - self.last_process_time >= self.wait_time) 


class STRBucket:
    def __init__(self):
        self.batch = []
        self.last_process_time = datetime.now()
        self.lock = asyncio.Lock()
        self.event = asyncio.Event()
        
    async def add(self, data):
        async with self.lock:
            self.batch.extend(data)
            self.event.set()

    async def get_batch(self):
        async with self.lock:
            batch = self.batch.copy()
            self.batch.clear()
            self.event.clear()
            return batch