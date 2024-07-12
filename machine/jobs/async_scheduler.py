import asyncio
import concurrent
import concurrent.futures
import threading
from typing import Set


class AsyncScheduler:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._start_background_loop, daemon=True).start()
        self._tasks: Set[concurrent.futures.Future] = set()

    def _start_background_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def schedule(self, coro) -> None:
        task = asyncio.run_coroutine_threadsafe(coro, self._loop)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def stop(self) -> None:
        concurrent.futures.wait(self._tasks)
        self._tasks.clear()
        self._loop.stop()
