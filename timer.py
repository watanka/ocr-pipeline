import asyncio

class Timer:
    def __init__(self, wait_time: int):
        self.wait_time = wait_time
        self.event = asyncio.Event()
        asyncio.create_task(self.timeit())

    async def timeit(self):
        print('timer 시작')
        countdown = 0
        while True:
            print('timer 대기 중')
            await asyncio.sleep(self.wait_time)
            if not self.event.is_set() and countdown >= self.wait_time:
                self.event.set()
                countdown = 0
            else:
                countdown += self.wait_time

    async def wait_for_event(self):
        while True:
            await self.event.wait()
            print('timer 이벤트 기다리고 있었다구!')
            self.event.clear()

async def main():
    timer = Timer(10)
    await timer.wait_for_event()




if __name__ == '__main__':
    asyncio.run(main())

