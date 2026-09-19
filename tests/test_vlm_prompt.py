"""The Arabic prompt builder. Importing vlm is cheap — mlx loads inside Captioner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_state import SceneState  # noqa: E402
from vlm import ARABIC_PROMPT, build_prompt  # noqa: E402


def test_no_context_falls_back_to_the_single_sentence_prompt():
    assert build_prompt(None) == ARABIC_PROMPT
    assert build_prompt("") == ARABIC_PROMPT
    assert build_prompt("   \n ") == ARABIC_PROMPT


def test_context_prompt_embeds_the_scene_and_asks_what_changed():
    state = SceneState()
    boxes = [{"x1": 0, "y1": 0, "x2": 10, "y2": 10, "label": "person",
              "label_ar": "شخص", "score": 0.9}]
    state.update(boxes, now=0.0)
    state.add_caption("شخص يجلس أمام مكتب.", now=1.0)

    prompt = build_prompt(state.caption_context(now=2.0))

    assert "شخص" in prompt                       # the detected object
    assert "شخص يجلس أمام مكتب." in prompt        # the previous caption
    assert "ما الذي تغيّر" in prompt               # asks for change, not inventory
    assert "بجملتين" in prompt                    # two sentences
    assert prompt != ARABIC_PROMPT


def test_braces_in_context_do_not_break_formatting():
    # str.format on attacker-ish text would raise; the template must survive it.
    assert "{oops}" in build_prompt("الأجسام الحالية: {oops}")
