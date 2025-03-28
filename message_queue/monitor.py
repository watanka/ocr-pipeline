import asyncio
from typing import Callable
from datetime import datetime
import uuid

from .bucket import BatchBucket, STDBucket, STRBucket
from common.schema import DetectionResponse
from common.logger import setup_logger, LOG_FORMATS

logger = setup_logger(
    'bucket_monitor',
    format_string=LOG_FORMATS['DETAILED']
)

class BucketMonitor:
    def __init__(self, bucket: BatchBucket, interval: float):
        self.bucket = bucket
        self.event = asyncio.Event()
        self.interval = interval

    # async def monitor(self, callback: Callable):
    #     logger.info("BucketMonitor 시작")
    #     while True:
    #         await asyncio.sleep(self.interval)
    #         logger.info("BucketMonitor 대기 중")
            

    #         messages = await self.bucket.get_batch()
            
    #         logger.info(f"버켓 안 배치 크기: {len(messages)}")
    #         if self.bucket.is_ready():
    #             messages = DetectionResponse(
    #                 id=str(uuid.uuid4())[:10], # TODO: 식별자 구분 필요
    #                 regions=messages,
    #                 result_image=None,
    #                 total_regions=len(messages)
    #             )
    #             self.bucket.batch.clear()
    #             self.bucket.last_process_time = datetime.now()
    #             logger.info(f"배치 크기 {len(messages)}개 처리 시작")
    #             await callback(messages)

            
    async def monitor(self, process_message: Callable, callback: Callable):
        logger.info(f"BucketMonitor [{self.bucket.name}] 시작")
        while True:
            await asyncio.sleep(self.interval)
            logger.info(f'BucketMonitor [{self.bucket.name}] 모니터링...')

            messages = await self.bucket.get_batch()

            logger.info(f'버켓 [{self.bucket.name}] 배치 크기: {len(messages)}')
            if self.bucket.is_ready():
                processed_messages = process_message(messages)
                self.bucket.batch.clear()
                self.bucket.last_process_time = datetime.now()
                logger.info(f'버켓 [{self.bucket.name}] 배치 크기 {len(messages)}개 처리 시작')
                await callback(processed_messages)



class STDBucketMonitor:
    def __init__(self, bucket: STDBucket, interval: float):
        self.bucket = bucket
        self.interval = interval
            
    async def monitor(self, process_message: Callable, callback: Callable):
        logger.info(f"STDBucketMonitor 시작")
        while True:
            await asyncio.sleep(self.interval)
            logger.info(f'STDBucketMonitor 모니터링...')

            if self.bucket.is_ready():
                messages = await self.bucket.get_batch()
                processed_messages = process_message(messages)
                self.bucket.last_process_time = datetime.now()
                logger.info(f'STDBucket 배치 크기 {len(messages)}개 처리 시작')
                await callback(processed_messages)



class STRBucketMonitor:
    def __init__(self, bucket: STRBucket, result_queue: asyncio.Queue):
        self.bucket = bucket
        self.queue = result_queue # STR 결과 큐

    async def monitor(self):
        logger.info(f"STRBucketMonitor 시작")
        while True:
            await self.bucket.event.wait()
            messages = await self.bucket.get_batch()
            logger.info(f'STRBucket 배치 크기: {len(messages)}')
            await self.queue.put(messages)
