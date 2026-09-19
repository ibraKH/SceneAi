"""RF-DETR Nano detection wrapper (PyTorch, MPS with CPU fallback)."""

import io
import logging

from PIL import Image

from labels import COCO_ID_TO_NAME, arabic

log = logging.getLogger("scene_ai.detector")

MODEL_NAME = "RF-DETR Nano"
# Frames are downscaled to this width before inference; smaller = faster.
INFER_WIDTH = 512


def _class_lookup():
    """dict of {coco_id: english_name}, preferring the table shipped with rfdetr.

    The module moved in rfdetr 1.9 (rfdetr.util -> rfdetr.assets), so try both.
    """
    table = None
    for module in ("rfdetr.assets.coco_classes", "rfdetr.util.coco_classes"):
        try:
            table = __import__(module, fromlist=["COCO_CLASSES"]).COCO_CLASSES
            break
        except Exception:
            continue

    if table is None:
        return dict(COCO_ID_TO_NAME)
    if isinstance(table, dict):
        return {int(k): v for k, v in table.items()}
    # some versions ship a list indexed by class id
    return dict(enumerate(table))


class Detector:
    def __init__(self, threshold: float = 0.45):
        self.threshold = threshold
        self.names = _class_lookup()
        self.device = self._pick_device()

        from rfdetr import RFDETRNano

        try:
            self.model = RFDETRNano(device=self.device)
        except TypeError:
            # older signatures don't accept device=
            self.model = RFDETRNano()
        log.info("RF-DETR Nano loaded on %s", self.device)

    @staticmethod
    def _pick_device() -> str:
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _fallback_to_cpu(self):
        if self.device == "cpu":
            return
        log.warning("MPS inference failed, falling back to CPU for detection")
        self.device = "cpu"
        from rfdetr import RFDETRNano

        try:
            self.model = RFDETRNano(device="cpu")
        except TypeError:
            self.model = RFDETRNano()

    def _predict(self, image: Image.Image):
        try:
            return self.model.predict(image, threshold=self.threshold)
        except TypeError:
            return self.model.predict(image)

    def infer(self, jpeg_bytes: bytes):
        """JPEG bytes -> (boxes, frame_width, frame_height).

        Boxes are dicts in the coordinate space of the *original* frame:
        {x1, y1, x2, y2, label, label_ar, score}
        """
        image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        ow, oh = image.size

        scale = 1.0
        if ow > INFER_WIDTH:
            scale = INFER_WIDTH / ow
            image = image.resize((INFER_WIDTH, max(1, round(oh * scale))))

        try:
            det = self._predict(image)
        except Exception:
            self._fallback_to_cpu()
            det = self._predict(image)

        boxes = []
        xyxy = getattr(det, "xyxy", None)
        if xyxy is None:
            return boxes, ow, oh

        confs = getattr(det, "confidence", None)
        class_ids = getattr(det, "class_id", None)
        inv = 1.0 / scale

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) * inv for v in xyxy[i])
            cid = int(class_ids[i]) if class_ids is not None else -1
            name = self.names.get(cid, f"class_{cid}")
            boxes.append(
                {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                    "label": name,
                    "label_ar": arabic(name),
                    "score": round(float(confs[i]) if confs is not None else 0.0, 3),
                }
            )
        return boxes, ow, oh

    def prewarm(self):
        """One dummy inference so the first real frame isn't paying compile cost."""
        buf = io.BytesIO()
        Image.new("RGB", (640, 480), (110, 110, 110)).save(buf, "JPEG", quality=70)
        self.infer(buf.getvalue())
