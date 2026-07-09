"""
form_standards.py

A small curated lookup of correct-form standards for the exercises this app
coaches -- canonical checkpoints, common faults, and adaptive modifications,
each with a cited source. Injected into the coaching prompts so the guidance
is grounded in recognized standards rather than only the model's own recall,
and so the advice can cite where it comes from.

This is deliberately a LOOKUP TABLE, not retrieval/RAG: with a bounded set of
exercises, a hand-curated table delivers grounded, citable coaching at a
fraction of the complexity of a vector store. Retrieval earns its place later,
for open-ended exercise requests and a large adaptive-fitness corpus (see the
three-tier plan in the project docs).

Sources are real, recognized references (ACSM = American College of Sports
Medicine; NCHPAD = National Center on Health, Physical Activity and
Disability). The specific cue wording here is a plain-language summary for
coaching, not a verbatim quotation.
"""

from __future__ import annotations

# Each entry: canonical checkpoints, the faults to watch for, adaptive
# modifications for limited mobility/balance, and a source to cite.
FORM_STANDARDS: dict[str, dict] = {
    "squat": {
        "match": ["squat", "bodyweight squat", "chair squat", "sit to stand"],
        "checkpoints": [
            "feet about shoulder-width apart, weight through the whole foot",
            "hips travel back and down as the knees bend, chest staying tall",
            "knees track in line with the feet (not caving inward)",
            "descend to a comfortable depth, ideally thighs near parallel to the floor",
        ],
        "faults": ["knees caving inward", "heels lifting", "rounding the lower back", "descending too fast"],
        "adaptive": "For balance or lower-limb limits, do it as a sit-to-stand from a sturdy chair, keeping a hand on a stable surface.",
        "source": "ACSM resistance-training guidance",
    },
    "wall push-up": {
        "match": ["push-up", "push up", "pushup", "wall push", "incline push"],
        "checkpoints": [
            "hands on the wall a bit wider than the shoulders, at chest height",
            "body in one straight line from head to heels",
            "elbows bend back at roughly 45 degrees, not flared straight out",
            "chest moves toward the wall under control, then press back",
        ],
        "faults": ["hips sagging or piking", "elbows flaring wide", "partial range of motion"],
        "adaptive": "The wall version lightens the load; move the feet closer to the wall to make it easier, farther to make it harder.",
        "source": "ACSM resistance-training guidance",
    },
    "seated shoulder press": {
        "match": ["shoulder press", "overhead press", "military press", "seated press"],
        "checkpoints": [
            "sit tall with the back supported, core gently braced",
            "start with hands at shoulder height, elbows under the wrists",
            "press straight up without arching the lower back",
            "stop short of locking out harshly; lower under control",
        ],
        "faults": ["arching the lower back", "pressing the weight forward instead of up", "shrugging the shoulders up to the ears"],
        "adaptive": "Seated with back support reduces balance and spine demand; use a light or no weight to start.",
        "source": "ACSM resistance-training guidance",
    },
    "standing calf raise": {
        "match": ["calf raise", "heel raise", "calf"],
        "checkpoints": [
            "stand tall near a stable surface for light balance support",
            "rise smoothly onto the balls of the feet",
            "pause briefly at the top, then lower the heels under control",
            "keep the ankles tracking straight, not rolling outward",
        ],
        "faults": ["bouncing", "rolling onto the outer edge of the foot", "short range of motion"],
        "adaptive": "Keep a hand on a wall or rail for balance; do it seated with light resistance if standing balance is limited.",
        "source": "NCHPAD adaptive strength guidance",
    },
    "chair-assisted lunge": {
        "match": ["lunge", "split squat", "chair lunge", "static lunge"],
        "checkpoints": [
            "one hand on a sturdy chair for balance",
            "step one foot forward into a split stance",
            "lower straight down so the front knee stays over the ankle",
            "keep the torso upright; press back up through the front foot",
        ],
        "faults": ["front knee drifting past the toes", "leaning the torso forward", "losing balance from too-narrow a stance"],
        "adaptive": "The chair gives balance support; reduce the range or hold the bottom briefly rather than full depth to start.",
        "source": "NCHPAD adaptive strength guidance",
    },
}


def lookup(exercise_name: str) -> dict | None:
    """Return the standards entry whose match-terms best fit the exercise
    name, or None if nothing matches. Case-insensitive substring match."""
    if not exercise_name:
        return None
    q = exercise_name.lower()
    for entry in FORM_STANDARDS.values():
        if any(term in q for term in entry["match"]):
            return entry
    return None


def as_prompt_context(exercise_name: str) -> str:
    """A compact, injectable block of grounded standards for the coaching
    prompts. Empty string if the exercise isn't in the table (the model then
    falls back to its own knowledge, which is fine for out-of-table asks)."""
    entry = lookup(exercise_name)
    if not entry:
        return ""
    return (
        "\n\nGrounded form standards for this exercise (base your coaching on these, and you may "
        f"cite the source):\n"
        f"- Correct checkpoints: {'; '.join(entry['checkpoints'])}.\n"
        f"- Common faults to watch for: {', '.join(entry['faults'])}.\n"
        f"- Adaptive modification: {entry['adaptive']}\n"
        f"- Source: {entry['source']}."
    )


if __name__ == "__main__":
    for name in ["bodyweight squat", "wall push-up", "shoulder press", "jumping jacks"]:
        hit = lookup(name)
        print(f"{name!r} -> {hit['source'] if hit else 'no match (falls back to model knowledge)'}")
