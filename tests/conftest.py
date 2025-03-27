import pytest
import asyncio
import logging

# pytest-asyncio 설정
pytest_plugins = ('pytest_asyncio',)

# 로깅 설정
logging.basicConfig(level=logging.INFO)

@pytest.fixture(scope="session")
def event_loop():
    """pytest-asyncio용 이벤트 루프 픽스처"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close() 