"""Outgoing live results prefer freshness without losing semantic messages."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_transport import LiveOutbox  # noqa: E402


def test_queued_detections_are_replaced_by_the_newest_result():
    async def scenario():
        outbox = LiveOutbox()
        outbox.put_detection("detection-1")
        outbox.put_detection("detection-2")
        outbox.put_detection("detection-3")
        return await outbox.get()

    assert asyncio.run(scenario()) == "detection-3"


def test_reliable_messages_are_ordered_and_not_replaced():
    async def scenario():
        outbox = LiveOutbox()
        await outbox.put_reliable("event-1")
        outbox.put_detection("detection-old")
        outbox.put_detection("detection-new")
        await outbox.put_reliable("caption-1")
        return [await outbox.get() for _ in range(3)]

    assert asyncio.run(scenario()) == ["event-1", "caption-1", "detection-new"]


def test_waiting_consumer_wakes_for_a_detection():
    async def scenario():
        outbox = LiveOutbox()
        waiter = asyncio.create_task(outbox.get())
        await asyncio.sleep(0)
        outbox.put_detection("fresh")
        return await asyncio.wait_for(waiter, timeout=0.1)

    assert asyncio.run(scenario()) == "fresh"
