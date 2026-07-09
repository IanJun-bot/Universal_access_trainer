"""
claude_review.py

Sends the Ollama-generated draft script to Claude for a review pass.
Claude checks the script against an accessibility checklist and returns
a revised, approved version — this is the "judgment" layer of the
pipeline, catching vague or unsafe language before it ever reaches TTS.

Requires: ANTHROPIC_API_KEY set in your environment (or a .env file,
          see README.md).
"""

import os
import anthropic

from ollama_draft import DRAFT_SYSTEM_PROMPT
from form_standards import as_prompt_context

# Model choice: claude-sonnet-5 is a good default for this — strong
# instruction-following and reasoning at reasonable cost/latency.
# Swap to claude-haiku-4-5-20251001 for faster/cheaper iteration while
# you're testing, or claude-opus-4-8 if you want the most careful pass.
REVIEW_MODEL = "claude-sonnet-5"

REVIEW_SYSTEM_PROMPT = """You are reviewing a verbal exercise script that will be read aloud
to a blind or low-vision listener with no visual reference. Your job is quality control,
not rewriting from scratch.

Check the draft against this checklist:
1. SPATIAL PRECISION — Flag and fix any vague spatial language ("in front of you," "like this,"
   "over there"). Replace with a plain relative direction (forward/back, left/right, up/down,
   toward/away from the body), a body-relative distance, or a tactile cue.
2. CLOCK POSITIONS MUST BE PAIRED, NEVER STANDALONE — Clock-face language ("3 o'clock") is a
   legitimate blind orientation-and-mobility convention, but not every listener was taught it.
   Find every clock-position reference in the draft. If it appears without a plain relative
   direction in the same phrase, add one -- e.g. "to 3 o'clock" alone must become "out to your
   right side, to about 3 o'clock." The script must be fully followable by someone who has
   never heard of clock positions, using only the plain-direction half of each pair.
3. PREFER FELT DISTANCE OVER ABSTRACT UNITS — Where the draft uses a raw unit ("six inches,"
   "two feet") and a body-relative equivalent would work just as well ("about the width of your
   hand," "one big step"), prefer the felt version. Abstract units aren't wrong, just a weaker
   default when a touch-based comparison is available.
4. NO VISUAL DEPENDENCE — Search line by line for: look, looks, looking, see, seeing, watch,
   watching, notice how it looks, appears, visually, as shown, as pictured, as illustrated, or
   any instruction assuming the listener can see a demo, mirror, or screen. Rewrite every
   instance using a plain direction, distance, or feel-based cue instead -- do not just soften
   the wording, replace the underlying assumption that the listener can see anything.
5. SAFETY — Flag anything that could cause injury without a spotter/visual check (e.g. balance-heavy
   moves near furniture, overhead loaded movements) and add a brief grounding or safety cue.
   Ensure the FINAL step includes a plain reminder to stop and rest if they feel sharp pain,
   dizziness, or loss of balance, and to check with a doctor or trainer if something feels
   unsafe. If it's missing, add it to step 10.
6. PACING — Make sure timing/count cues are explicit enough to follow by ear alone
   (e.g. "hold for a slow count of three," not just "hold briefly").
7. STRUCTURE — Keep it to 10 numbered steps, each 1-2 sentences, natural to read aloud at a
   conversational pace. No title, no introduction, no closing remarks.

Return ONLY the final, approved 10-step script — no preamble, no meta-commentary, no
"here is the revised version" framing. If the draft already meets the checklist, return it
essentially as-is with only minor polish.
"""

# Generous ceiling. claude-sonnet-5 uses extended thinking, and thinking +
# output must together fit under max_tokens. You are billed only for tokens
# actually generated, so a high ceiling costs nothing on normal completions
# -- it just stops the thinking phase from crowding out the answer and
# truncating it (stop_reason="max_tokens" with no text block). See
# _complete_text for the retry that catches the rare remaining case.
MAX_TOKENS = 8192

# Total attempts = MAX_RETRIES + 1. The empty-output failure is
# non-deterministic and almost always clears on the next attempt.
MAX_RETRIES = 2


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Set it as an environment variable "
            "or add it to a .env file (see README.md)."
        )
    return anthropic.Anthropic(api_key=api_key)


def _complete_text(client: anthropic.Anthropic, *, model: str, system: str, user_content: str) -> str:
    """
    Call Claude and return the concatenated text blocks, retrying on the
    intermittent failure where extended thinking leaves no usable text --
    either the response truncates at max_tokens or comes back with no text
    block at all. Silently returning "" from here is what produced the
    original "generates nothing, no error shown" bug, so instead we retry
    the whole call and, if every attempt fails, raise a clear error the UI
    can surface rather than handing back an empty script.
    """
    last_stop = None
    for _ in range(MAX_RETRIES + 1):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        last_stop = response.stop_reason
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        # A max_tokens stop means the script is truncated (cut off mid-step),
        # which is unusable even if some text came back -- treat it as a miss.
        if text and response.stop_reason != "max_tokens":
            return text
    raise RuntimeError(
        f"Claude returned no usable script after {MAX_RETRIES + 1} attempts "
        f"(last stop_reason={last_stop}). This is usually transient -- try again."
    )


def review_script(draft_script: str, exercise_context: str = "", model: str = REVIEW_MODEL) -> str:
    """
    Send a draft script to Claude for an accessibility/safety review pass.

    Args:
        draft_script: raw output from ollama_draft.generate_draft()
        exercise_context: optional extra context to pass through (disability
            specifics, equipment, fitness level)
        model: Claude model string to use

    Returns:
        Approved/revised script text, ready for TTS.
    """
    user_content = f"Draft script to review:\n\n{draft_script}"
    if exercise_context:
        user_content += f"\n\nContext: {exercise_context}"

    return _complete_text(_get_client(), model=model, system=REVIEW_SYSTEM_PROMPT, user_content=user_content)


def draft_with_claude(exercise_name: str, extra_context: str = "", model: str = REVIEW_MODEL) -> str:
    """
    Generate the approved script directly via Claude, skipping the local
    Ollama draft step entirely. Reuses the exact same drafting instructions
    as the Ollama path (DRAFT_SYSTEM_PROMPT), so this demonstrates the same
    prompt architecture running on a production model -- the "Use Claude
    API?" toggle for the Audio Coach tab.

    Args:
        exercise_name: e.g. "bodyweight squat"
        extra_context: optional free text, e.g. "beginner, no equipment"
        model: Claude model string to use

    Returns:
        Approved script text, ready for TTS.
    """
    user_prompt = f"Exercise: {exercise_name}\n"
    if extra_context:
        user_prompt += f"Additional context: {extra_context}\n"
    user_prompt += as_prompt_context(exercise_name)  # grounded form standards, if known
    user_prompt += "\nWrite the 10-step verbal script now."

    return _complete_text(_get_client(), model=model, system=DRAFT_SYSTEM_PROMPT, user_content=user_prompt)


if __name__ == "__main__":
    sample_draft = (
        "1. Stand like this near a chair.\n"
        "2. Put your arms out in front of you.\n"
        "3. Bend down a bit.\n"
    )
    print(review_script(sample_draft))
