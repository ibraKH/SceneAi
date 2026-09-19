"""Qwen2.5-VL captioning via mlx-vlm (Apple MLX, 4-bit)."""

import inspect
import io
import logging
import os

from PIL import Image

log = logging.getLogger("scene_ai.vlm")

# --- To switch models on OOM, change this one line (or export VLM_MODEL_ID): ---
DEFAULT_MODEL_ID = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
DEFAULT_MODEL_REVISION = "46d4cf06a06ffc1a766c214174f9cbed2f45bcab"
# ------------------------------------------------------------------------------

MODEL_ID = os.environ.get("VLM_MODEL_ID", DEFAULT_MODEL_ID)
MODEL_REVISION = os.environ.get(
    "VLM_MODEL_REVISION",
    DEFAULT_MODEL_REVISION if MODEL_ID == DEFAULT_MODEL_ID else "",
) or None
SMALLER_MODEL_ID = "mlx-community/Qwen2-VL-2B-Instruct-4bit"

# Used when the caller has no scene context yet (prewarm, first frame, empty scene).
ARABIC_PROMPT = "صف ما تراه في هذه الصورة بجملة واحدة قصيرة."

# With context, the model is told what the detector found, how long each object has
# been there, what just changed, and what it said last — then asked for the situation
# rather than an inventory. {context} is the block from SceneState.caption_context().
ARABIC_CONTEXT_PROMPT = """معلومات عن المشهد من نظام الكشف:
{context}

صف ما يحدث الآن في المشهد وما الذي تغيّر عن الوصف السابق، بجملتين قصيرتين بالعربية.
لا تكرر قائمة الأجسام، بل صف الموقف والحركة. إن لم يتغيّر شيء فاذكر ذلك باختصار."""


def build_prompt(context: str | None) -> str:
    """Context-aware Arabic prompt, falling back to the single-sentence one."""
    if not context or not context.strip():
        return ARABIC_PROMPT
    return ARABIC_CONTEXT_PROMPT.format(context=context.strip())

OOM_HELP = f"""
================================ VLM OUT OF MEMORY ================================
Loading/running {MODEL_ID} ran out of unified memory.

Fix (one line change): open vlm.py and set

    DEFAULT_MODEL_ID = "{SMALLER_MODEL_ID}"

Or without editing anything, run:

    VLM_MODEL_ID={SMALLER_MODEL_ID} ./run.sh

Then close other heavy apps (Chrome tabs, Docker, Xcode) and start again.
===================================================================================
"""


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        k in text
        for k in ("out of memory", "oom", "metal", "insufficient memory", "cannot allocate")
    )


class Captioner:
    """Blocking captioner. Call from a worker thread, never the event loop."""

    def __init__(self, model_id: str = MODEL_ID, revision: str | None = MODEL_REVISION):
        self.model_id = model_id
        self.revision = revision
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template

        self._generate = generate
        self._apply_chat_template = apply_chat_template

        try:
            self.model, self.processor = load(model_id, revision=revision)
        except Exception as exc:
            if _is_oom(exc):
                log.error(OOM_HELP)
            raise

        self.config = self._load_config(model_id)
        self._gen_params = set(inspect.signature(generate).parameters)

    def _load_config(self, model_id: str):
        try:
            from mlx_vlm.utils import load_config

            return load_config(model_id, revision=self.revision)
        except Exception:
            cfg = getattr(self.model, "config", None)
            if cfg is None:
                return {}
            return cfg

    def _format_prompt(self, prompt: str) -> str:
        try:
            return self._apply_chat_template(
                self.processor, self.config, prompt, num_images=1
            )
        except TypeError:
            # older signature without num_images
            return self._apply_chat_template(self.processor, self.config, prompt)

    def _call_generate(self, formatted: str, image: Image.Image, max_tokens: int):
        kwargs = {"max_tokens": max_tokens, "verbose": False}
        if "temperature" in self._gen_params:
            kwargs["temperature"] = 0.0
        elif "temp" in self._gen_params:
            kwargs["temp"] = 0.0

        # mlx-vlm has moved the image argument around across releases; try the
        # keyword forms it has used, then the two positional orders.
        attempts = []
        if "image" in self._gen_params:
            attempts.append(lambda: self._generate(
                self.model, self.processor, formatted, image=[image], **kwargs))
        if "images" in self._gen_params:
            attempts.append(lambda: self._generate(
                self.model, self.processor, formatted, images=[image], **kwargs))
        attempts.append(lambda: self._generate(
            self.model, self.processor, formatted, [image], **kwargs))
        attempts.append(lambda: self._generate(
            self.model, self.processor, [image], formatted, **kwargs))

        last = None
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:
                last = exc
        raise RuntimeError(f"mlx-vlm generate() signature not recognised: {last}")

    @staticmethod
    def _as_text(result) -> str:
        text = getattr(result, "text", result)
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        return str(text).strip()

    # max_tokens is the main latency/quality tradeoff in the whole caption lane:
    # 48 bought ~1.0-1.7 s and one clipped sentence; 128 fits the two sentences the
    # context prompt asks for, at roughly proportional cost. Lower it first if the
    # caption period feels sluggish.
    def caption(self, jpeg_bytes: bytes, context: str | None = None,
                prompt: str | None = None, max_tokens: int = 128) -> str:
        """Caption one JPEG. With `context`, describes what changed, not just what is there."""
        prompt = prompt if prompt is not None else build_prompt(context)
        image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        # Keep the vision tower cheap: Qwen2.5-VL tokenises by area.
        image.thumbnail((448, 448))
        try:
            result = self._call_generate(self._format_prompt(prompt), image, max_tokens)
        except Exception as exc:
            if _is_oom(exc):
                log.error(OOM_HELP)
            raise
        return self._as_text(result)

    def prewarm(self):
        buf = io.BytesIO()
        Image.new("RGB", (448, 448), (120, 120, 120)).save(buf, "JPEG", quality=70)
        self.caption(buf.getvalue(), max_tokens=4)
