"""
claude_vision.py

Sends an exercise-form photo to Claude's vision API instead of the local
Ollama vision model. Used by the "Analyze with Claude instead of local
model" toggle in the Form Checker tab, to demonstrate the same prompt
running unchanged on a production model.
"""

import os
import base64
import anthropic

from form_standards import as_prompt_context

# Any current Claude model handles image input. Sonnet is a good default
# balance of quality, latency, and cost for this use case.
VISION_MODEL = "claude-sonnet-5"

FORM_CHECK_PROMPT = """You are analyzing a photo of a person performing the exercise: {exercise}.

Give exactly 3 short, text-based corrective captions about their form.
Rules:
- Do not use auditory language like "listen" or "hear" -- captions must stand alone as text.
- Keep each caption under 15 words.
- Be specific about body position (joint angles, alignment, weight distribution).
- If the form looks correct, say so in the first caption and use the remaining two for reinforcement.
- Format as a numbered list, nothing else.
- If the photo does not show enough to assess form confidently -- bad angle, cropped body,
  blurry, poor lighting, or it doesn't look like the stated exercise -- say so plainly as the
  first caption (e.g. "Can't assess form: photo doesn't show your knees or hips") instead of
  guessing. Do not invent corrections for body positions you can't actually see.
"""


SEQUENCE_CHECK_PROMPT = """You are analyzing {n} sequential frames, in chronological order, sampled
evenly from a video of one repetition of the exercise: {exercise}.

Before anything else, decide: can you clearly see a person performing this exercise in
these frames? If there is no person visible, the person is mostly out of frame, or the frames
show something other than the stated exercise, your first caption MUST state that you can't
assess the movement and why -- never describe or correct a movement you cannot actually see.

Assess the MOVEMENT across the frames -- not just one pose. Consider the descent/ascent path,
control and tempo, and where in the rep the form changes (e.g. "form is good at the start but
your knees cave inward at the bottom of the rep").

Give exactly 3 short, text-based corrective captions about their movement.
Rules:
- Do not use auditory language like "listen" or "hear" -- captions must stand alone as text.
- Keep each caption under 20 words.
- Be specific about body position and WHERE IN THE REP it happens (early, middle, bottom, rising).
- If the movement looks correct throughout, say so in the first caption and use the remaining
  two for reinforcement.
- Format as a numbered list, nothing else.
- If the frames do not show enough to assess the movement confidently -- bad angle, cropped
  body, blurry, poor lighting, or it doesn't look like the stated exercise -- say so plainly
  as the first caption instead of guessing. Do not invent corrections you can't actually see.
"""


def _media_type_for(image_bytes: bytes) -> str:
    """Sniff the image type from its header bytes so Claude gets the right media_type."""
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/jpeg"  # reasonable default


def check_form_with_claude(image_bytes: bytes, exercise_name: str = "exercise", model: str = VISION_MODEL) -> str:
    """
    Analyze a single exercise-form photo via Claude's vision API.

    Args:
        image_bytes: raw bytes of the uploaded image (jpg/png)
        exercise_name: name of the exercise being performed, for context
        model: Claude model string to use

    Returns:
        Numbered list of corrective captions, as text.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Set it as an environment variable "
            "or add it to a .env file (see README.md)."
        )

    client = anthropic.Anthropic(api_key=api_key)
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    media_type = _media_type_for(image_bytes)

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_image}},
                {"type": "text", "text": FORM_CHECK_PROMPT.format(exercise=exercise_name) + as_prompt_context(exercise_name)},
            ],
        }],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    return text.strip()


def check_form_sequence_with_claude(frames: list[bytes], exercise_name: str = "exercise", model: str = VISION_MODEL) -> str:
    """
    Analyze an ordered frame sequence from one repetition via Claude's
    vision API -- multiple image blocks in a single message, chronological.

    Args:
        frames: chronological JPEG frame bytes (from video_frames.extract_frames)
        exercise_name: name of the exercise being performed, for context
        model: Claude model string to use

    Returns:
        Numbered list of corrective captions, as text.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Set it as an environment variable "
            "or add it to a .env file (see README.md)."
        )

    client = anthropic.Anthropic(api_key=api_key)

    content: list = []
    for frame in frames:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": _media_type_for(frame),
                "data": base64.b64encode(frame).decode("utf-8"),
            },
        })
    content.append({
        "type": "text",
        "text": SEQUENCE_CHECK_PROMPT.format(n=len(frames), exercise=exercise_name) + as_prompt_context(exercise_name),
    })

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": content}],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    return text.strip()
