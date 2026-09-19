"""IOUTracker and SceneState. No models, no network, no clock dependence."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_state import IOUTracker, SceneState, iou, is_feminine_ar  # noqa: E402


def box(x1, y1, x2, y2, label="person", label_ar="شخص", score=0.9):
    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "label": label, "label_ar": label_ar, "score": score,
    }


# --------------------------------------------------------------- iou helper

def test_iou_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # 10x10 boxes sharing a 5x10 strip: 50 / (100 + 100 - 50)
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


# ------------------------------------------------------------------ tracker

def test_id_is_stable_while_the_object_drifts():
    tracker = IOUTracker()
    ids, _, _ = tracker.update([box(0, 0, 100, 100)], now=0.0)
    first = ids[0]

    # Ten frames of small drift; the id must never change.
    for frame in range(1, 11):
        ids, born, retired = tracker.update([box(frame * 3, 0, 100 + frame * 3, 100)],
                                            now=float(frame))
        assert ids == [first]
        assert born == [] and retired == []

    assert tracker.tracks[first].hits == 11


def test_a_jump_beyond_the_iou_threshold_starts_a_new_track():
    tracker = IOUTracker()
    ids, _, _ = tracker.update([box(0, 0, 100, 100)], now=0.0)
    moved, born, _ = tracker.update([box(400, 400, 500, 500)], now=1.0)

    assert moved != ids
    assert len(born) == 1


def test_two_objects_keep_separate_ids():
    tracker = IOUTracker()
    ids, _, _ = tracker.update([box(0, 0, 100, 100), box(300, 0, 400, 100)], now=0.0)
    assert len(set(ids)) == 2

    again, _, _ = tracker.update([box(5, 0, 105, 100), box(305, 0, 405, 100)], now=1.0)
    assert again == ids


def test_different_classes_never_match_each_other():
    tracker = IOUTracker()
    ids, _, _ = tracker.update([box(0, 0, 100, 100, "person", "شخص")], now=0.0)
    # Same place, different class: a new track, not a relabelled old one.
    chair, born, _ = tracker.update([box(0, 0, 100, 100, "chair", "كرسي")], now=1.0)

    assert chair != ids
    assert born[0].label == "chair"


def test_track_expires_after_max_misses():
    tracker = IOUTracker(max_misses=3)
    tracker.update([box(0, 0, 100, 100)], now=0.0)

    for frame in range(1, 3):
        _, _, retired = tracker.update([], now=float(frame))
        assert retired == []          # still within its grace period

    _, _, retired = tracker.update([], now=3.0)
    assert len(retired) == 1
    assert tracker.tracks == {}


def test_a_brief_miss_does_not_break_the_id():
    tracker = IOUTracker(max_misses=5)
    ids, _, _ = tracker.update([box(0, 0, 100, 100)], now=0.0)

    tracker.update([], now=1.0)       # dropped for one frame
    again, born, _ = tracker.update([box(0, 0, 100, 100)], now=2.0)

    assert again == ids
    assert born == []


def test_ids_are_never_reused_after_expiry():
    tracker = IOUTracker(max_misses=2)
    ids, _, _ = tracker.update([box(0, 0, 100, 100)], now=0.0)
    tracker.update([], now=1.0)
    tracker.update([], now=2.0)
    assert tracker.tracks == {}

    # Same class, same pixels, but it is a different object now.
    fresh, _, _ = tracker.update([box(0, 0, 100, 100)], now=3.0)
    assert fresh[0] != ids[0]


def test_misses_reset_on_a_rematch():
    tracker = IOUTracker(max_misses=3)
    ids, _, _ = tracker.update([box(0, 0, 100, 100)], now=0.0)
    tracker.update([], now=1.0)
    tracker.update([], now=2.0)
    tracker.update([box(0, 0, 100, 100)], now=3.0)

    assert tracker.tracks[ids[0]].misses == 0
    # Having been rematched, it must survive another two empty frames.
    _, _, retired = tracker.update([], now=4.0)
    assert retired == []


# --------------------------------------------------------------- scene state

def test_update_attaches_track_ids_to_the_boxes_in_place():
    state = SceneState()
    boxes = [box(0, 0, 100, 100), box(300, 0, 400, 100)]
    state.update(boxes, now=0.0)

    assert all("track_id" in b for b in boxes)
    assert boxes[0]["track_id"] != boxes[1]["track_id"]


def test_appeared_event_on_first_sight():
    state = SceneState()
    events = state.update([box(0, 0, 100, 100)], now=0.0)

    assert [e.kind for e in events] == ["appeared"]
    assert events[0].label_ar == "شخص"
    assert events[0].track_id is not None
    assert events[0].text_ar() == "ظهر شخص"


def test_no_events_while_nothing_changes():
    state = SceneState()
    state.update([box(0, 0, 100, 100)], now=0.0)
    assert state.update([box(2, 0, 102, 100)], now=1.0) == []


def test_disappeared_event_after_the_track_expires():
    state = SceneState(max_misses=2)
    state.update([box(0, 0, 100, 100)], now=0.0)
    assert state.update([], now=1.0) == []

    events = state.update([], now=2.0)
    assert [e.kind for e in events] == ["disappeared"]
    assert events[0].text_ar() == "اختفى شخص"


def test_count_changed_when_a_present_class_gains_a_member():
    state = SceneState()
    state.update([box(0, 0, 100, 100)], now=0.0)
    events = state.update([box(0, 0, 100, 100), box(300, 0, 400, 100)], now=1.0)

    kinds = [e.kind for e in events]
    assert "appeared" in kinds and "count_changed" in kinds
    assert next(e for e in events if e.kind == "count_changed").track_id is None


def test_no_count_changed_on_the_first_member_of_a_class():
    # 0 -> 1 is already fully described by "appeared".
    state = SceneState()
    events = state.update([box(0, 0, 100, 100)], now=0.0)
    assert [e.kind for e in events] == ["appeared"]


def test_events_deque_is_bounded():
    state = SceneState(max_misses=1)
    for frame in range(120):
        # Alternate present/absent so events keep firing.
        state.update([box(0, 0, 100, 100)] if frame % 2 == 0 else [], now=float(frame))

    assert len(state.events) <= 50
    assert state.event_seq > 50      # the running counter is not bounded


def test_recent_events_filters_by_age():
    state = SceneState()
    state.update([box(0, 0, 100, 100)], now=0.0)
    state.update([box(600, 600, 700, 700, "chair", "كرسي")], now=100.0)

    assert len(state.recent_events(seconds=10.0, now=100.0)) == 1
    assert len(state.recent_events(seconds=200.0, now=100.0)) == 2


# ------------------------------------------------------------ prompt summary

def test_summary_is_arabic_and_reports_counts_and_age():
    state = SceneState()
    state.update([box(0, 0, 100, 100), box(300, 0, 400, 100)], now=0.0)
    summary = state.summary_for_prompt(now=12.0)

    assert "شخص" in summary
    assert "×2" in summary
    assert "منذ 12 ثانية" in summary


def test_summary_uses_minutes_past_sixty_seconds():
    state = SceneState()
    state.update([box(0, 0, 100, 100)], now=0.0)
    assert "منذ 2 دقيقة" in state.summary_for_prompt(now=125.0)


def test_summary_when_the_scene_is_empty():
    assert SceneState().summary_for_prompt(now=0.0) == "لا توجد أجسام مكتشفة حالياً."


def test_summary_orders_by_descending_count():
    state = SceneState()
    state.update(
        [
            box(0, 0, 100, 100, "chair", "كرسي"),
            box(300, 0, 400, 100),
            box(500, 0, 600, 100),
        ],
        now=0.0,
    )
    summary = state.summary_for_prompt(now=0.0)
    assert summary.index("شخص") < summary.index("كرسي")


# ------------------------------------------------------------------ captions

def test_last_caption_round_trips():
    state = SceneState()
    assert state.last_caption() is None

    state.add_caption("شخص يجلس أمام الحاسوب.", now=0.0)
    state.add_caption("الشخص غادر الغرفة.", now=5.0)
    assert state.last_caption() == "الشخص غادر الغرفة."


def test_captions_deque_is_bounded():
    state = SceneState()
    for i in range(40):
        state.add_caption(f"وصف {i}", now=float(i))
    assert len(state.captions) == 10


def test_new_events_since_caption_resets_when_a_caption_lands():
    state = SceneState()
    state.update([box(0, 0, 100, 100)], now=0.0)
    assert state.new_events_since_caption() == 1

    state.add_caption("شخص في الغرفة.", now=1.0)
    assert state.new_events_since_caption() == 0

    state.update([box(0, 0, 100, 100), box(300, 0, 400, 100)], now=2.0)
    assert state.new_events_since_caption() > 0


def test_event_created_during_caption_inference_remains_unacknowledged():
    state = SceneState()
    state.update([box(0, 0, 100, 100)], now=0.0)

    context, event_watermark = state.snapshot_caption_context(now=1.0)
    assert "ظهر شخص" in context

    # Detection continues while the captioner is working with the snapshot.
    state.update(
        [box(0, 0, 100, 100), box(300, 0, 400, 100, "chair", "كرسي")],
        now=2.0,
    )
    state.add_caption("شخص وكرسي في الغرفة.", now=3.0, event_watermark=event_watermark)

    assert state.new_events_since_caption() == state.event_seq - event_watermark
    assert state.new_events_since_caption() > 0


def test_caption_context_carries_objects_changes_and_previous_caption():
    state = SceneState()
    state.update([box(0, 0, 100, 100)], now=0.0)
    state.add_caption("شخص يجلس أمام مكتب.", now=1.0)
    context = state.caption_context(seconds=30.0, now=2.0)

    assert "الأجسام الحالية:" in context
    assert "شخص" in context
    assert "التغيّرات الأخيرة:" in context and "ظهر شخص" in context
    assert "الوصف السابق: شخص يجلس أمام مكتب." in context


def test_caption_context_omits_empty_sections():
    context = SceneState().caption_context(now=0.0)
    assert "التغيّرات الأخيرة:" not in context
    assert "الوصف السابق:" not in context


# ------------------------------------------------------- arabic agreement

@pytest.mark.parametrize("label_ar", ["ثلاجة", "حقيبة يد", "لوحة مفاتيح", "دراجة نارية"])
def test_feminine_labels_are_detected(label_ar):
    assert is_feminine_ar(label_ar)


@pytest.mark.parametrize("label_ar", ["شخص", "كرسي", "حاسوب", "جهاز تحكم", ""])
def test_masculine_labels_are_detected(label_ar):
    assert not is_feminine_ar(label_ar)


def test_event_text_agrees_with_a_feminine_noun():
    state = SceneState(max_misses=1)
    fridge = box(0, 0, 100, 100, "refrigerator", "ثلاجة")
    appeared = state.update([fridge], now=0.0)[0]
    disappeared = state.update([], now=1.0)[0]

    assert appeared.text_ar() == "ظهرت ثلاجة"
    assert disappeared.text_ar() == "اختفت ثلاجة"


def test_event_text_agrees_with_a_masculine_noun():
    state = SceneState(max_misses=1)
    appeared = state.update([box(0, 0, 100, 100)], now=0.0)[0]
    disappeared = state.update([], now=1.0)[0]

    assert appeared.text_ar() == "ظهر شخص"
    assert disappeared.text_ar() == "اختفى شخص"


def test_event_to_dict_is_json_shaped():
    state = SceneState()
    payload = state.update([box(0, 0, 100, 100)], now=3.5)[0].to_dict()

    assert payload["kind"] == "appeared"
    assert payload["label"] == "person"
    assert payload["label_ar"] == "شخص"
    assert payload["timestamp"] == 3.5
    assert payload["text_ar"] == "ظهر شخص"
    assert isinstance(payload["track_id"], int)
