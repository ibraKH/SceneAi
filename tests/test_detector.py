"""Coordinate rescaling in detector.py, with the model mocked out.

`Detector.__init__` imports rfdetr and pulls 349 MB of weights off disk, so these
tests build the object with `object.__new__` and inject a fake model. That is the
only way to exercise the rescale arithmetic without a GPU.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import detector as detector_mod  # noqa: E402
from detector import INFER_WIDTH, Detector  # noqa: E402


class FakeDetections:
    """Stands in for supervision.Detections."""

    def __init__(self, xyxy, class_id=None, confidence=None):
        self.xyxy = np.array(xyxy, dtype=float)
        self.class_id = np.array(class_id, dtype=int) if class_id is not None else None
        self.confidence = (
            np.array(confidence, dtype=float) if confidence is not None else None
        )


class FakeModel:
    """Records the image it was handed, returns canned boxes in inference space."""

    def __init__(self, detections):
        self.detections = detections
        self.seen_sizes = []

    def predict(self, image, threshold=None):
        self.seen_sizes.append(image.size)
        return self.detections


def make_detector(detections) -> Detector:
    det = object.__new__(Detector)
    det.threshold = 0.45
    det.device = "cpu"
    det.names = {1: "person", 62: "chair"}
    det.model = FakeModel(detections)
    return det


def jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (90, 110, 90)).save(buf, "JPEG", quality=70)
    return buf.getvalue()


def test_frame_is_downscaled_to_the_inference_width():
    det = make_detector(FakeDetections([[0, 0, 10, 10]], [1], [0.9]))
    det.infer(jpeg(640, 480))

    assert det.model.seen_sizes == [(INFER_WIDTH, 384)]   # 480 * 512/640


def test_boxes_are_rescaled_back_into_original_frame_coordinates():
    # A box covering the whole 512-wide inference image must come back covering
    # the whole 640-wide transport frame.
    det = make_detector(FakeDetections([[0, 0, 512, 384]], [1], [0.9]))
    boxes, w, h = det.infer(jpeg(640, 480))

    assert (w, h) == (640, 480)
    assert boxes[0]["x1"] == 0.0
    assert boxes[0]["y1"] == 0.0
    assert boxes[0]["x2"] == pytest.approx(640.0, abs=0.1)
    assert boxes[0]["y2"] == pytest.approx(480.0, abs=0.1)


def test_rescale_uses_the_exact_inverse_scale():
    det = make_detector(FakeDetections([[100, 50, 200, 150]], [1], [0.9]))
    boxes, _, _ = det.infer(jpeg(640, 360))

    inv = 640 / INFER_WIDTH            # 1.25
    assert boxes[0]["x1"] == pytest.approx(100 * inv, abs=0.05)
    assert boxes[0]["y1"] == pytest.approx(50 * inv, abs=0.05)
    assert boxes[0]["x2"] == pytest.approx(200 * inv, abs=0.05)
    assert boxes[0]["y2"] == pytest.approx(150 * inv, abs=0.05)


def test_small_frames_are_not_upscaled_and_pass_through_unscaled():
    det = make_detector(FakeDetections([[10, 20, 30, 40]], [1], [0.9]))
    boxes, w, h = det.infer(jpeg(320, 240))

    assert det.model.seen_sizes == [(320, 240)]     # untouched
    assert (w, h) == (320, 240)
    assert (boxes[0]["x1"], boxes[0]["y1"]) == (10.0, 20.0)
    assert (boxes[0]["x2"], boxes[0]["y2"]) == (30.0, 40.0)


def test_labels_are_translated_to_arabic():
    det = make_detector(FakeDetections([[0, 0, 50, 50], [60, 0, 110, 50]], [1, 62], [0.9, 0.5]))
    boxes, _, _ = det.infer(jpeg(640, 480))

    assert [b["label"] for b in boxes] == ["person", "chair"]
    assert [b["label_ar"] for b in boxes] == ["شخص", "كرسي"]


def test_unknown_class_id_gets_a_placeholder_label():
    det = make_detector(FakeDetections([[0, 0, 50, 50]], [999], [0.9]))
    boxes, _, _ = det.infer(jpeg(640, 480))

    assert boxes[0]["label"] == "class_999"
    assert boxes[0]["label_ar"] == "class_999"       # falls back to the English name


def test_scores_are_rounded_to_three_places():
    det = make_detector(FakeDetections([[0, 0, 50, 50]], [1], [0.4954321]))
    boxes, _, _ = det.infer(jpeg(640, 480))
    assert boxes[0]["score"] == 0.495


def test_no_detections_returns_an_empty_list_with_frame_size():
    det = make_detector(FakeDetections(np.empty((0, 4)), [], []))
    boxes, w, h = det.infer(jpeg(640, 480))

    assert boxes == []
    assert (w, h) == (640, 480)


def test_missing_xyxy_is_tolerated():
    class Empty:
        xyxy = None

    det = make_detector(Empty())
    boxes, w, h = det.infer(jpeg(640, 480))

    assert boxes == []
    assert (w, h) == (640, 480)


def test_predict_falls_back_when_the_model_rejects_the_threshold_kwarg():
    class OldSignatureModel(FakeModel):
        def predict(self, image, threshold=None):
            if threshold is not None:
                raise TypeError("predict() got an unexpected keyword argument 'threshold'")
            return self.detections

    det = make_detector(FakeDetections([[0, 0, 512, 384]], [1], [0.9]))
    det.model = OldSignatureModel(det.model.detections)
    boxes, _, _ = det.infer(jpeg(640, 480))

    assert len(boxes) == 1


def test_class_lookup_returns_a_usable_id_to_name_map():
    names = detector_mod._class_lookup()

    assert isinstance(names, dict)
    assert names.get(1) == "person"
