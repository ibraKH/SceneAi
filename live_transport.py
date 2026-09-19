"""Small freshness-first outgoing transport for the live WebSocket."""

from __future__ import annotations

import asyncio


class LiveOutbox:
    """Keep only the newest detection while delivering control messages in order."""

    def __init__(self, reliable_capacity: int = 64):
        self._reliable: asyncio.Queue[str] = asyncio.Queue(maxsize=reliable_capacity)
        self._latest_detection: str | None = None
        self._wake = asyncio.Event()

    def put_detection(self, message: str) -> None:
        self._latest_detection = message
        self._wake.set()

    async def put_reliable(self, message: str) -> None:
        await self._reliable.put(message)
        self._wake.set()

    async def get(self) -> str:
        while True:
            try:
                return self._reliable.get_nowait()
            except asyncio.QueueEmpty:
                pass

            if self._latest_detection is not None:
                message = self._latest_detection
                self._latest_detection = None
                return message

            self._wake.clear()
            # Re-check after clearing so a producer cannot wake us between the
            # empty checks above and the wait below.
            if not self._reliable.empty() or self._latest_detection is not None:
                self._wake.set()
                continue
            await self._wake.wait()
