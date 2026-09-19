"""The world model: what is present, when it appeared, and what changed.

Everything here is pure Python and side-effect free apart from its own state, so
it can be unit-tested without loading a single model. `server.py` owns one
`SceneState` per WebSocket connection.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

# Two boxes of the same class overlapping by this much are the same object.
# Low enough to survive the jitter RF-DETR produces frame to frame, high enough
# that two people standing side by side don't merge.
IOU_THRESHOLD = 0.3
# ~1.5 s at 10 frames/s: long enough to ride out a few missed detections without
# the id churning, short enough that a departure is reported promptly.
MAX_MISSES = 15

MAX_EVENTS = 50
MAX_CAPTIONS = 10

Box = tuple[float, float, float, float]


@dataclass
class Track:
    """One object followed across frames, with a stable id."""

    id: int
    label: str
    label_ar: str
    box: Box
    score: float
    first_seen: float
    last_seen: float
    hits: int = 1
    misses: int = 0

    def age(self, now: float | None = None) -> float:
        """Seconds since this object was first seen."""
        return (time.time() if now is None else now) - self.first_seen


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two xyxy boxes. 0.0 when they don't overlap."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IOUTracker:
    """Greedy IOU tracker: stable integer ids, no dependencies, no learning.

    Matching is constrained to boxes of the same class. Without that, a chair
    and the person sitting on it swap ids the moment their boxes overlap.
    """

    def __init__(self, iou_threshold: float = IOU_THRESHOLD, max_misses: int = MAX_MISSES):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def _new_id(self) -> int:
        # Monotonic and never reused: a recycled id would make a new object look
        # like the return of an old one to anything downstream.
        track_id = self._next_id
        self._next_id += 1
        return track_id

    def update(
        self, detections: list[dict], now: float | None = None
    ) -> tuple[list[int], list[Track], list[Track]]:
        """Advance one frame.

        `detections` are detector boxes: {x1, y1, x2, y2, label, label_ar, score}.
        Returns (ids parallel to `detections`, tracks born now, tracks retired now).
        """
        now = time.time() if now is None else now
        det_boxes: list[Box] = [
            (float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])) for d in detections
        ]

        # Score every legal (track, detection) pair, then take them best-first.
        pairs = []
        for track_id, track in self.tracks.items():
            for di, det in enumerate(detections):
                if det.get("label") != track.label:
                    continue
                overlap = iou(track.box, det_boxes[di])
                if overlap >= self.iou_threshold:
                    pairs.append((overlap, track_id, di))
        pairs.sort(key=lambda p: p[0], reverse=True)

        assigned_ids: list[int | None] = [None] * len(detections)
        matched_tracks: set[int] = set()
        for _, track_id, di in pairs:
            if track_id in matched_tracks or assigned_ids[di] is not None:
                continue
            matched_tracks.add(track_id)
            assigned_ids[di] = track_id
            track = self.tracks[track_id]
            track.box = det_boxes[di]
            track.score = float(detections[di].get("score", 0.0))
            track.last_seen = now
            track.hits += 1
            track.misses = 0

        born: list[Track] = []
        born_ids: set[int] = set()
        for di, det in enumerate(detections):
            if assigned_ids[di] is not None:
                continue
            track = Track(
                id=self._new_id(),
                label=det.get("label", ""),
                label_ar=det.get("label_ar", det.get("label", "")),
                box=det_boxes[di],
                score=float(det.get("score", 0.0)),
                first_seen=now,
                last_seen=now,
            )
            self.tracks[track.id] = track
            assigned_ids[di] = track.id
            born.append(track)
            born_ids.add(track.id)

        retired: list[Track] = []
        for track_id in list(self.tracks):
            if track_id in matched_tracks or track_id in born_ids:
                continue
            track = self.tracks[track_id]
            track.misses += 1
            if track.misses >= self.max_misses:
                retired.append(self.tracks.pop(track_id))

        # Every detection now owns an id, so this list is parallel to `detections`.
        return [int(i) for i in assigned_ids if i is not None], born, retired

    def active(self) -> list[Track]:
        """Tracks currently considered present, oldest first."""
        return sorted(self.tracks.values(), key=lambda t: t.first_seen)


def is_feminine_ar(label_ar: str) -> bool:
    """Whether an Arabic label takes feminine verb agreement.

    A taa marbuta on the head noun is the reliable marker: ثلاجة, حقيبة يد and
    لوحة مفاتيح are feminine, جهاز تحكم and شخص are not. Only the first word is
    checked, since the rest is a modifier ("حقيبة يد" agrees with حقيبة).
    """
    head = label_ar.strip().split(" ")[0] if label_ar.strip() else ""
    return head.endswith("ة")


@dataclass
class SceneEvent:
    """Something that changed. `track_id` is None for count_changed."""

    kind: str  # "appeared" | "disappeared" | "count_changed"
    label: str
    label_ar: str
    timestamp: float
    track_id: int | None = None

    def text_ar(self) -> str:
        feminine = is_feminine_ar(self.label_ar)
        if self.kind == "appeared":
            return f"{'ظهرت' if feminine else 'ظهر'} {self.label_ar}"
        if self.kind == "disappeared":
            return f"{'اختفت' if feminine else 'اختفى'} {self.label_ar}"
        return f"تغيّر عدد {self.label_ar}"

    def to_dict(self) -> dict:
        # text_ar travels on the wire so the frontend never builds Arabic strings.
        return {
            "kind": self.kind,
            "label": self.label,
            "label_ar": self.label_ar,
            "timestamp": round(self.timestamp, 3),
            "track_id": self.track_id,
            "text_ar": self.text_ar(),
        }


@dataclass
class Caption:
    text: str
    timestamp: float


def _duration_ar(seconds: float) -> str:
    """Arabic phrase for how long something has been present."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"منذ {seconds} ثانية"
    minutes = seconds // 60
    return f"منذ {minutes} دقيقة"


class SceneState:
    """Tracks, recent events and recent captions for one connection."""

    def __init__(self, iou_threshold: float = IOU_THRESHOLD, max_misses: int = MAX_MISSES):
        self.tracker = IOUTracker(iou_threshold=iou_threshold, max_misses=max_misses)
        self.events: deque[SceneEvent] = deque(maxlen=MAX_EVENTS)
        self.captions: deque[Caption] = deque(maxlen=MAX_CAPTIONS)
        # Monotonic count of every event ever emitted, so the caption trigger can
        # ask "anything new since last time?" without scanning the bounded deque.
        self.event_seq = 0
        self._event_seq_at_last_caption = 0
        self._counts: dict[str, int] = {}

    def update(self, boxes: list[dict], now: float | None = None) -> list[SceneEvent]:
        """Run the tracker over one frame's boxes and emit what changed.

        Each box dict gains a stable `track_id` in place.
        """
        now = time.time() if now is None else now
        ids, born, retired = self.tracker.update(boxes, now=now)
        for box, track_id in zip(boxes, ids):
            box["track_id"] = track_id

        events: list[SceneEvent] = []
        for track in born:
            events.append(
                SceneEvent("appeared", track.label, track.label_ar, now, track.id)
            )
        for track in retired:
            events.append(
                SceneEvent("disappeared", track.label, track.label_ar, now, track.id)
            )

        # count_changed covers the case appear/disappear can't express on its own:
        # a class that was present before and after, at a different count.
        counts: dict[str, int] = {}
        labels_ar: dict[str, str] = {}
        for track in self.tracker.tracks.values():
            counts[track.label] = counts.get(track.label, 0) + 1
            labels_ar[track.label] = track.label_ar
        for label, count in counts.items():
            before = self._counts.get(label, 0)
            if before and before != count:
                events.append(
                    SceneEvent("count_changed", label, labels_ar[label], now, None)
                )
        self._counts = counts

        self.events.extend(events)
        self.event_seq += len(events)
        return events

    def summary_for_prompt(self, now: float | None = None) -> str:
        """Arabic object counts and how long each class has been present."""
        now = time.time() if now is None else now
        grouped: dict[str, list[Track]] = {}
        for track in self.tracker.active():
            grouped.setdefault(track.label, []).append(track)
        if not grouped:
            return "لا توجد أجسام مكتشفة حالياً."

        parts = []
        for label, tracks in sorted(
            grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])
        ):
            oldest = min(t.first_seen for t in tracks)
            parts.append(
                f"{tracks[0].label_ar} ×{len(tracks)} ({_duration_ar(now - oldest)})"
            )
        return "، ".join(parts)

    def recent_events(self, seconds: float = 10.0, now: float | None = None) -> list[SceneEvent]:
        now = time.time() if now is None else now
        cutoff = now - seconds
        return [e for e in self.events if e.timestamp >= cutoff]

    def last_caption(self) -> str | None:
        return self.captions[-1].text if self.captions else None

    def add_caption(
        self,
        text: str,
        now: float | None = None,
        event_watermark: int | None = None,
    ) -> None:
        now = time.time() if now is None else now
        self.captions.append(Caption(text=text, timestamp=now))
        # A caption only acknowledges events included in its prompt. Detection
        # can advance event_seq while the VLM is running in another thread.
        watermark = self.event_seq if event_watermark is None else event_watermark
        self._event_seq_at_last_caption = max(
            self._event_seq_at_last_caption, min(watermark, self.event_seq)
        )

    def new_events_since_caption(self) -> int:
        """How many events have landed since the last caption was recorded."""
        return self.event_seq - self._event_seq_at_last_caption

    def caption_context(self, seconds: float = 10.0, now: float | None = None) -> str:
        """The Arabic block handed to the VLM: what's here, what changed, what we said."""
        now = time.time() if now is None else now
        lines = [f"الأجسام الحالية: {self.summary_for_prompt(now=now)}"]

        recent = self.recent_events(seconds=seconds, now=now)
        if recent:
            # Newest first, de-duplicated, capped — the prompt is not a log file.
            seen: list[str] = []
            for event in reversed(recent):
                text = event.text_ar()
                if text not in seen:
                    seen.append(text)
                if len(seen) >= 6:
                    break
            lines.append("التغيّرات الأخيرة: " + "، ".join(seen))

        previous = self.last_caption()
        if previous:
            lines.append(f"الوصف السابق: {previous}")
        return "\n".join(lines)

    def snapshot_caption_context(
        self, seconds: float = 10.0, now: float | None = None
    ) -> tuple[str, int]:
        """Atomically capture a prompt context and the events it contains."""
        return self.caption_context(seconds=seconds, now=now), self.event_seq
