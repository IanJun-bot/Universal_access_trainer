"""
vision_form_checker.py

Uses a local Ollama vision-language model to analyze a single uploaded
photo of exercise form and return short, text-only corrective captions —
built for deaf users who can't rely on verbal coaching cues, and who
shouldn't have to watch a live video stream mid-set.

Requires: Ollama running locally, and the vision model pulled:
    ollama pull qwen2.5vl:7b
"""

import ollama
from model_manager import switch_to
from form_standards import as_prompt_context

VISION_MODEL = "qwen2.5vl:7b"

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


def check_form(image_bytes: bytes, exercise_name: str = "exercise", model: str = VISION_MODEL) -> str:
    """
    Analyze a single exercise-form photo and return 3 text corrective captions.

    Args:
        image_bytes: raw bytes of the uploaded image (jpg/png)
        exercise_name: name of the exercise being performed, for context
        model: Ollama vision model tag to use

    Returns:
        Numbered list of corrective captions, as text.
    """
    switch_to(model)  # unload any other resident Ollama model first -- keeps VRAM usage sequential

    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": FORM_CHECK_PROMPT.format(exercise=exercise_name) + as_prompt_context(exercise_name),
                "images": [image_bytes],
            }],
        )
    except Exception as e:
        raise RuntimeError(
            f"Vision model call failed -- is '{model}' pulled? Try: ollama pull {model}. "
            f"Original error: {e}"
        ) from e

    return response["message"]["content"].strip()


def check_form_sequence(frames: list[bytes], exercise_name: str = "exercise", model: str = VISION_MODEL) -> str:
    """
    Analyze an ordered sequence of frames from one repetition and return
    3 movement-level corrective captions.

    Args:
        frames: chronological JPEG frame bytes (from video_frames.extract_frames)
        exercise_name: name of the exercise being performed, for context
        model: Ollama vision model tag to use

    Returns:
        Numbered list of corrective captions, as text.
    """
    switch_to(model)

    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": SEQUENCE_CHECK_PROMPT.format(n=len(frames), exercise=exercise_name) + as_prompt_context(exercise_name),
                "images": frames,
            }],
        )
    except Exception as e:
        raise RuntimeError(
            f"Vision model call failed -- is '{model}' pulled? Try: ollama pull {model}. "
            f"Original error: {e}"
        ) from e

    return response["message"]["content"].strip()


if __name__ == "__main__":
    with open("test_photo.jpg", "rb") as f:
        print(check_form(f.read(), exercise_name="squat"))
