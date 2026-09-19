"""Real-time Arabic scene dashboard: RF-DETR detection + Qwen2.5-VL captions."""

import asyncio
import json
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

import vlm as vlm_mod
from detector import MODEL_NAME as DET_MODEL_NAME
from live_transport import LiveOutbox
from scene_state import SceneState
from security import is_local_origin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scene_ai")

HERE = Path(__file__).parent
INDEX = HERE / "static" / "index.html"

DET_THRESHOLD = 0.45

# Captions are change-triggered, not periodic: a new scene event wakes the lane,
# but never sooner than MIN (the VLM would starve the detector) and never later
# than MAX (a still scene should still get refreshed).
MIN_CAPTION_INTERVAL_S = 3.0
MAX_CAPTION_INTERVAL_S = 12.0
# How often the idle caption lane re-checks for new events. Cheap: it is a bare
# integer comparison on the event loop.
CAPTION_POLL_S = 0.25

# One thread each so detection and captioning never fight over the GPU mid-kernel.
DET_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="det")
VLM_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm")


class Runtime:
    def __init__(self):
        self.detector = None
        self.captioner = None
        self.ready = False
        self.error: str | None = None
        self.status = "بدء التشغيل…"


RT = Runtime()


def load_everything():
    """Blocking: import, load and prewarm both models. Runs in a worker thread."""
    t0 = time.perf_counter()
    try:
        RT.status = "تحميل نموذج الكشف…"
        log.info("Loading detection model (%s)…", DET_MODEL_NAME)
        from detector import Detector

        RT.detector = Detector(threshold=DET_THRESHOLD)
        RT.detector.prewarm()
        log.info("Detection ready  (%s, %.1fs)", RT.detector.device, time.perf_counter() - t0)

        RT.status = "تحميل النموذج اللغوي البصري…"
        t1 = time.perf_counter()
        log.info("Loading VLM (%s)…", vlm_mod.MODEL_ID)
        RT.captioner = vlm_mod.Captioner()
        RT.captioner.prewarm()
        log.info("VLM ready  (%.1fs)", time.perf_counter() - t1)

        RT.ready = True
        RT.status = "جاهز"
        log.info("=" * 58)
        log.info("Ready. Open http://localhost:8000")
        log.info("=" * 58)
    except Exception as exc:
        RT.error = f"{type(exc).__name__}: {exc}"
        RT.status = "فشل تحميل النماذج"
        log.exception("Model loading failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kicked off in the background so the page (and its loading screen) serves
    # immediately; /ws refuses connections until RT.ready flips.
    task = asyncio.create_task(asyncio.to_thread(load_everything))
    yield
    task.cancel()
    DET_POOL.shutdown(wait=False, cancel_futures=True)
    VLM_POOL.shutdown(wait=False, cancel_futures=True)


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(INDEX)


@app.get("/health")
async def health():
    return JSONResponse(
        {
            "ready": RT.ready,
            "error": RT.error,
            "status": RT.status,
            "detector": DET_MODEL_NAME,
            "vlm": vlm_mod.MODEL_ID,
        }
    )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    if not is_local_origin(ws.headers.get("origin")):
        log.warning("Rejected non-local WebSocket Origin: %r", ws.headers.get("origin"))
        await ws.close(code=1008)
        return
    await ws.accept()

    if not RT.ready:
        await ws.send_text(
            json.dumps(
                {
                    "type": "status",
                    "state": "error" if RT.error else "loading",
                    "message": RT.error or RT.status,
                }
            )
        )
        await ws.close()
        return

    await ws.send_text(
        json.dumps(
            {
                "type": "status",
                "state": "ready",
                "detector": f"{DET_MODEL_NAME} ({RT.detector.device})",
                "vlm": vlm_mod.MODEL_ID.split("/")[-1],
            }
        )
    )

    loop = asyncio.get_running_loop()
    latest = {"frame": None, "seq": 0}
    outbox = LiveOutbox(reliable_capacity=64)
    # One world model per connection, owned entirely by the event loop: the
    # detection worker writes it, the caption worker reads it, and neither runs
    # inside a thread pool, so no lock is needed.
    scene = SceneState()

    async def sender():
        while True:
            await ws.send_text(await outbox.get())

    async def emit(payload: dict):
        message = json.dumps(payload, ensure_ascii=False)
        if payload.get("type") == "detection":
            outbox.put_detection(message)
        else:
            await outbox.put_reliable(message)

    async def detection_worker():
        seen = -1
        durations: deque[float] = deque(maxlen=15)
        while True:
            if latest["seq"] == seen or latest["frame"] is None:
                await asyncio.sleep(0.005)
                continue
            seen = latest["seq"]
            frame = latest["frame"]

            t0 = time.perf_counter()
            try:
                boxes, fw, fh = await loop.run_in_executor(
                    DET_POOL, RT.detector.infer, frame
                )
            except Exception:
                log.exception("Detection failed")
                await asyncio.sleep(0.5)
                continue
            durations.append(time.perf_counter() - t0)

            # Stamps a stable track_id onto every box and tells us what changed.
            events = scene.update(boxes)

            total = sum(durations)
            fps = round(len(durations) / total, 1) if total > 0 else 0.0
            await emit({"type": "detection", "boxes": boxes, "fps": fps, "w": fw, "h": fh})
            if events:
                await emit({"type": "events", "events": [e.to_dict() for e in events]})

    async def caption_worker():
        # Both windows are measured from when the *previous* caption started, not
        # from when it finished, so a 1.5 s inference no longer pushes the whole
        # cadence 1.5 s later every round (ARCHITECTURE.md §7).
        anchor = loop.time()
        while True:
            now = loop.time()
            earliest, forced = anchor + MIN_CAPTION_INTERVAL_S, anchor + MAX_CAPTION_INTERVAL_S

            if now < earliest:                     # too soon, whatever happened
                await asyncio.sleep(earliest - now)
                continue
            if scene.new_events_since_caption() == 0 and now < forced:
                await asyncio.sleep(min(CAPTION_POLL_S, forced - now))
                continue
            if latest["frame"] is None:
                await asyncio.sleep(CAPTION_POLL_S)
                continue

            frame = latest["frame"]  # always the newest; older frames are dropped
            context, event_watermark = scene.snapshot_caption_context()
            anchor = loop.time()     # the next window opens from here, not from the end
            t0 = time.perf_counter()
            try:
                text = await loop.run_in_executor(
                    VLM_POOL, RT.captioner.caption, frame, context
                )
            except Exception as exc:
                log.exception("Captioning failed")
                await emit(
                    {
                        "type": "caption",
                        "text": f"تعذّر توليد الوصف: {type(exc).__name__}",
                        "latency_ms": 0,
                        "error": True,
                    }
                )
                continue                            # the MIN window paces the retry

            latency = int((time.perf_counter() - t0) * 1000)
            if text:
                scene.add_caption(
                    text, event_watermark=event_watermark
                )                                   # feeds the next caption's context
                await emit({"type": "caption", "text": text, "latency_ms": latency})

    tasks = [
        asyncio.create_task(sender()),
        asyncio.create_task(detection_worker()),
        asyncio.create_task(caption_worker()),
    ]

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data:
                latest["frame"] = data
                latest["seq"] += 1
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("WebSocket receive loop error")
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("Client disconnected")
