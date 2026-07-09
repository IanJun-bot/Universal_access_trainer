"""
ollama_draft.py

Talks to a locally running Ollama server to produce a first-pass,
step-by-step verbal script describing an exercise, written for a
blind or low-vision listener (no visual references, precise spatial
language only).

Requires: Ollama running locally (default http://localhost:11434)
          and a model already pulled, e.g.:
              ollama pull llama3.1:8b-instruct-q4_K_M
"""

import requests

from form_standards import as_prompt_context

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1:8b-instruct-q4_K_M"

DRAFT_SYSTEM_PROMPT = """You write verbal exercise scripts for blind and low-vision listeners.
The listener cannot see a demonstration, so every instruction must rely on:
- body-relative distance, described through touch, not abstract units where possible
  ("about the width of your hand," "one big step back," "arm's length away" rather than
  "six inches" or "two feet" -- a felt reference is more universal than a unit of measurement)
- tactile/proprioceptive cues (what a correct position should FEEL like)
- clear sequencing and pacing (how long to hold, how many counts per rep)

Clock-position language ("raise your arm to 3 o'clock") is a real, standard technique from
blind orientation-and-mobility training -- it is learned through tactile watches, not sight,
so it is not inherently visual. But not every listener was taught it, and you cannot assume
they were. So: you may use clock positions, but ONLY paired with a plain relative direction
in the very same phrase, never alone. Correct: "raise your arm out to your right side, to
about 3 o'clock." Wrong: "raise your arm to 3 o'clock" with nothing else. The plain direction
(forward/back, left/right, up/down, toward/away from your body) must be enough on its own for
someone who has never heard of clock positions to follow the instruction correctly.

Never assume the listener can see a mirror, a screen, a trainer, or their own body position.
Do not use any of these words or their variants, even in passing: look, looks, looking, see,
seeing, watch, watching, notice how it looks, appears, visually, as shown, as pictured,
as illustrated, like this, like so, over there. If you catch yourself about to write one of
these, replace it with a plain direction, distance, or feel-based cue instead.

Output exactly 10 numbered steps and nothing else -- no title, no introduction, no closing
remarks. Start directly with "1.". Keep each step to 1-2 sentences. Include a brief safety
note where genuinely relevant (e.g. balance risk, joint strain) as part of the relevant step,
not as a separate step. The final step (step 10) must always include a plain safety reminder
to stop right away and rest if they feel sharp pain, dizziness, or lose their balance, and to
check with a doctor or trainer if an exercise ever feels unsafe for their body.
"""


def generate_draft(
    exercise_name: str,
    extra_context: str = "",
    model: str = DEFAULT_MODEL,
    host: str = OLLAMA_URL,
) -> str:
    """
    Ask the local Ollama model for a first-pass 10-step verbal script.

    Args:
        exercise_name: e.g. "bodyweight squat"
        extra_context: optional free text, e.g. "user uses a wheelchair,
            focus on upper body only" or "beginner, no equipment"
        model: Ollama model tag to use
        host: Ollama chat endpoint

    Returns:
        Raw draft script text from the model.
    """
    user_prompt = f"Exercise: {exercise_name}\n"
    if extra_context:
        user_prompt += f"Additional context: {extra_context}\n"
    user_prompt += as_prompt_context(exercise_name)  # grounded form standards, if known
    user_prompt += "\nWrite the 10-step verbal script now."

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }

    try:
        resp = requests.post(host, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            "Could not reach Ollama. Is it running? Try `ollama serve` "
            "or check that the desktop app is open."
        ) from e
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Ollama returned an error. Is '{model}' pulled? "
            f"Try: ollama pull {model}"
        ) from e

    data = resp.json()
    return data["message"]["content"].strip()


if __name__ == "__main__":
    draft = generate_draft("bodyweight squat", extra_context="beginner, no equipment")
    print(draft)
