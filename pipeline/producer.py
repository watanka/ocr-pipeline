import logging
import time
from confluent_kafka import Producer
from confluent_kafka.error import KafkaError
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_producer(bootstrap_servers, batch_size, linger_ms, retries=5, retry_interval=5):
    """Kafka Producer 생성 with 재시도 로직"""
    for attempt in range(retries):
        try:
            producer_config = {
                'bootstrap.servers': bootstrap_servers,
                'batch.size': batch_size,
                'linger.ms': linger_ms,
                'request.timeout.ms': 5000,
                'message.timeout.ms': 10000,
                'retry.backoff.ms': 1000,
                'client.id': 'pipeline-producer'
            }
            
            producer = Producer(producer_config)
            
            # 연결 테스트를 위한 메타데이터 요청
            producer.list_topics(timeout=5)
            logger.info("Successfully connected to Kafka")
            return producer
            
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(retry_interval)
            else:
                raise HTTPException(status_code=500, detail=f"Failed to connect to Kafka after {retries} attempts")
