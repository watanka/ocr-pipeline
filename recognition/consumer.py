from confluent_kafka import Consumer, KafkaError
import json
import requests
import logging
from recognition_model import process_recognition_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRConsumer:
    def __init__(self, 
                 bootstrap_servers: str = 'kafka:29092',
                 topic: str = 'ocr_topic',
                 group_id: str = 'ocr_pipeline_group',):
        
        self.topic = topic
        
        # Consumer 설정
        consumer_config = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
            'max.poll.interval.ms': 300000,  # 최대 처리 시간 (5분)
            'session.timeout.ms': 180000,    # 세션 타임아웃 (3분)
            'fetch.max.bytes': 52428800,     # 최대 fetch 크기 (50MB)
        }
        
        self.consumer = Consumer(consumer_config)
        self.consumer.subscribe([self.topic])
        
    def process_message(self, detection_result: dict) -> None:
        """개별 메시지 처리"""
        try:

            recognition_result = process_recognition_model(detection_result)
            logger.info(f"Recognition result: {recognition_result}")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def run(self):
        """Consumer 실행"""
        logger.info("Starting OCR Consumer...")
        
        try:
            while True:
                msg = self.consumer.poll(1.0)  # 1초 동안 메시지 대기
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.info('Reached end of partition')
                    else:
                        logger.error(f'Error: {msg.error()}')
                    continue
                
                try:
                    # 메시지 디코딩
                    detection_result = json.loads(msg.value().decode('utf-8'))
                    logger.info(f"Received detection result: {detection_result}")
                    
                    # 메시지 처리
                    self.process_message(detection_result)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message: {e}")
                except Exception as e:
                    logger.error(f"Failed to process message: {e}")
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Consumer 종료"""
        self.consumer.close()
        logger.info("Consumer closed")

if __name__ == "__main__":
    consumer = OCRConsumer()
    consumer.run()
