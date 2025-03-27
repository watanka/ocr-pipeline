import pytest
import asyncio
import logging
import subprocess
import time
from typing import Dict, Any
from pathlib import Path

from message_queue import MessageQueueFactory

# 로깅 설정
logger = logging.getLogger(__name__)

# 테스트 설정
RABBITMQ_URL = "amqp://admin:admin@localhost:5672/"
REDIS_URL = "redis://localhost:6379"
TEST_QUEUE = "test_queue"

def wait_for_service(host: str, port: int, timeout: int = 30) -> bool:
    """서비스가 준비될 때까지 대기"""
    import socket
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (ConnectionRefusedError, socket.timeout):
            if time.time() - start_time > timeout:
                return False
            time.sleep(1)

@pytest.fixture(scope="session", autouse=True)
async def message_queues():
    """메시지 큐 서버 시작/종료 픽스처"""
    logger.info("Starting message queue servers...")
    
    # 현재 스크립트의 디렉토리 경로
    current_dir = Path(__file__).parent.absolute()
    compose_file = current_dir / "docker-compose.test.yml"
    
    # Docker Compose 실행
    try:
        subprocess.run(
            ["docker-compose", "-f", str(compose_file), "up", "-d"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Message queue servers starting...")
        
        # 서비스가 준비될 때까지 대기
        if wait_for_service("localhost", 5672) and wait_for_service("localhost", 6379):
            logger.info("Services are accessible, waiting for RabbitMQ initialization...")
            # RabbitMQ가 완전히 초기화될 때까지 충분히 대기
            time.sleep(15)
            logger.info("Message queue servers are ready")
        else:
            raise TimeoutError("Services failed to start in time")
        
        yield
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start message queues: {e.stderr}")
        raise
    finally:
        # 서버 종료
        subprocess.run(
            ["docker-compose", "-f", str(compose_file), "down"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Message queue servers stopped")

# 메시지 수신 콜백
async def message_callback(message: Dict[str, Any]):
    logger.info(f"Received message: {message}")

@pytest.mark.asyncio
async def test_rabbitmq():
    """RabbitMQ 테스트"""
    logger.info("Testing RabbitMQ...")
    
    # RabbitMQ 클라이언트 생성
    mq = MessageQueueFactory.create("rabbitmq", url=RABBITMQ_URL)
    
    try:
        # 연결
        await mq.connect()
        
        # 구독 태스크 시작
        subscriber_task = asyncio.create_task(
            mq.subscribe(TEST_QUEUE, message_callback)
        )
        
        # 잠시 대기하여 구독이 설정되도록 함
        await asyncio.sleep(1)
        
        # 테스트 메시지 발행
        test_message = {
            "id": "test_1",
            "content": "Hello, RabbitMQ!",
            "timestamp": "2024-03-19 12:00:00"
        }
        await mq.publish(TEST_QUEUE, test_message)
        
        # 메시지가 처리될 때까지 대기
        await asyncio.sleep(2)
        
        # 구독 태스크 취소
        subscriber_task.cancel()
        try:
            await subscriber_task
        except asyncio.CancelledError:
            pass
        
        # 연결 해제
        await mq.disconnect()
        logger.info("RabbitMQ test completed successfully")
        
    except Exception as e:
        logger.error(f"RabbitMQ test failed: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_redis():
    """Redis 테스트"""
    logger.info("Testing Redis...")
    
    # Redis 클라이언트 생성
    mq = MessageQueueFactory.create("redis", url=REDIS_URL)
    
    try:
        # 연결
        await mq.connect()
        
        # 구독 태스크 시작
        subscriber_task = asyncio.create_task(
            mq.subscribe(TEST_QUEUE, message_callback)
        )
        
        # 잠시 대기하여 구독이 설정되도록 함
        await asyncio.sleep(1)
        
        # 테스트 메시지 발행
        test_message = {
            "id": "test_1",
            "content": "Hello, Redis!",
            "timestamp": "2024-03-19 12:00:00"
        }
        await mq.publish(TEST_QUEUE, test_message)
        
        # 메시지가 처리될 때까지 대기
        await asyncio.sleep(2)
        
        # 구독 태스크 취소
        subscriber_task.cancel()
        try:
            await subscriber_task
        except asyncio.CancelledError:
            pass
        
        # 연결 해제
        await mq.disconnect()
        logger.info("Redis test completed successfully")
        
    except Exception as e:
        logger.error(f"Redis test failed: {str(e)}")
        raise 