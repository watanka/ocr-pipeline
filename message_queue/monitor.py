import asyncio
from .bucket import BatchBucket
from common.logger import setup_logger, LOG_FORMATS
from typing import Callable
from datetime import datetime
logger = setup_logger(
    'bucket_monitor',
    format_string=LOG_FORMATS['DETAILED']
)

class BucketMonitor:
    def __init__(self, bucket: BatchBucket, interval: float):
        self.bucket = bucket
        self.event = asyncio.Event()
        self.interval = interval

    async def monitor(self, callback: Callable):
        logger.info("BucketMonitor 시작")
        while True:
            await asyncio.sleep(self.interval)
            logger.info("BucketMonitor 대기 중")
            

            messages = await self.bucket.get_batch()
            logger.info(f"버켓 안 배치 크기: {len(messages)}")
            if self.bucket.is_ready():
                self.bucket.batch.clear()
                self.bucket.last_process_time = datetime.now()
                logger.info(f"배치 크기 {len(messages)}개 처리 시작")
                await callback(messages)

            
