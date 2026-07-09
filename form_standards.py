"""
form_standards.py

A small curated lookup of correct-form standards for the exercises this app
coaches -- canonical checkpoints, common faults, and adaptive modifications,
each with specific, publicly accessible references. Injected into the coaching
prompts so the guidance is grounded in recognized published standards rather
than only the model's own recall.

This is deliberately a LOOKUP TABLE, not retrieval/RAG: with a bounded set of
exercises, a hand-curated table delivers grounded, citable coaching at a
fraction of the complexity of a vector store. Retrieval over full source
documents earns its place later, for open-ended exercise requests and a large
adaptive-fitness corpus (see the three-tier plan in the project docs).

ON THE CITATIONS (verified July 9, 2026): every entry lists the specific
published references its checkpoints were reconciled against, with URLs.
The checkpoint wording is still a plain-language coaching summary -- not a
verbatim quotation -- but it was checked line-by-line for consistency with
the listed references, and wording was adjusted where the references differed
(e.g. wall push-up hand placement follows the NIA instructions exactly).
Reference set:
  - ACE (American Council on Exercise) Exercise Library -- written per-exercise
    technique pages (bodyweight squat, seated overhead press, forward lunge).
  - NIA (National Institute on Aging, NIH), "Workout to Go" -- written
    step-by-step instructions for wall push-ups, overhead arm raises (seated
    option), and toe stands, all with chair-support balance guidance.
  - ACSM (American College of Sports Medicine), "Resistance Training for
    Health and Fitness" consumer brochure (2013) -- program-level guidance and
    the stop-if-pain safety directive that every generated script carries.
  - NCHPAD (National Center on Health, Physical Activity and Disability) --
    controlled-movement technique guidance and adaptive/seated resources.
"""

from __future__ import annotations

# Program-level references that apply to every exercise in the table.
GENERAL_SOURCES: list[dict] = [
    {
        "org": "ACSM",
        "title": "Resistance Training for Health and Fitness (consumer brochure, 2013)",
        "url": "https://www.prescriptiontogetactive.com/static/pdfs/resistance-training-ACSM.pdf",
        "note": "program-level guidance; directs users to stop any exercise that causes pain and seek medical advice",
    },
    {
        "org": "NCHPAD",
        "title": "High-Intensity Weight Training for People with Disabilities",
        "url": "https://www.nchpad.org/resources/high-intensity-weight-training-for-people-with-disabilities/",
        "note": "proper technique and controlled movement: no bouncing, throwing, or jerking; slow, controlled repetitions",
    },
]

# Each entry: canonical checkpoints, the faults to watch for, adaptive
# modifications for limited mobility/balance, and the specific references the
# checkpoints were reconciled against (see module docstring).
FORM_STANDARDS: dict[str, dict] = {
    "squat": {
        "match": ["squat", "bodyweight squat", "chair squat", "sit to stand"],
        "checkpoints": [
            "feet about shoulder-width or slightly wider, toes turned slightly out, weight spread through the whole foot",
            "hips travel back and then down as the knees bend, chest staying up and out",
            "knees track in line with the feet, roughly over the second toe (not caving inward)",
            "descend to a comfortable depth, ideally thighs near parallel to the floor -- stop if the heels lift or the back starts to round",
        ],
        "faults": ["knees caving inward", "heels lifting", "rounding the lower back", "descending too fast"],
        "adaptive": "For balance or lower-limb limits, do it as a sit-to-stand from a sturdy chair, keeping a hand on a stable surface.",
        "sources": [
            {
                "org": "ACE",
                "title": "Exercise Library: Bodyweight Squat",
                "url": "https://www.acefitness.org/resources/everyone/exercise-library/135/bodyweight-squat/",
            },
        ],
    },
    "wall push-up": {
        "match": ["push-up", "push up", "pushup", "wall push", "incline push"],
        "checkpoints": [
            "stand facing the wall, a little farther than arm's length away, feet shoulder-width apart",
            "palms flat on the wall at shoulder height and about shoulder-width apart",
            "body in one straight line from head to heels, feet staying flat on the floor",
            "bend the elbows to lower the chest toward the wall under control, hold a moment, then press back until the arms are straight",
        ],
        "faults": ["hips sagging or piking", "heels lifting off the floor", "partial range of motion"],
        "adaptive": "The wall version lightens the load; move the feet closer to the wall to make it easier, farther to make it harder.",
        "sources": [
            {
                "org": "NIA (NIH)",
                "title": "Workout to Go: Wall Push-Up",
                "url": "https://www.goaging.org/wp-content/uploads/2015/11/workout_to_go.pdf",
            },
        ],
    },
    "seated shoulder press": {
        "match": ["shoulder press", "overhead press", "military press", "seated press"],
        "checkpoints": [
            "sit tall with the back supported, feet firmly on the floor, core gently braced",
            "start with hands at shoulder height, wrists neutral and elbows under the wrists",
            "press straight up without arching the lower back",
            "keep a soft bend at the top rather than slamming the elbows straight; lower under control",
        ],
        "faults": ["arching the lower back", "pressing the weight forward instead of up", "shrugging the shoulders up to the ears"],
        "adaptive": "Seated with back support reduces balance and spine demand; the NIA version can be done seated with light or no weight, and NCHPAD demonstrates a wheelchair-based version.",
        "sources": [
            {
                "org": "ACE",
                "title": "Exercise Library: Seated Overhead Press",
                "url": "https://www.acefitness.org/resources/everyone/exercise-library/45/seated-overhead-press/",
            },
            {
                "org": "NIA (NIH)",
                "title": "Workout to Go: Overhead Arm Raise (standing or seated)",
                "url": "https://www.goaging.org/wp-content/uploads/2015/11/workout_to_go.pdf",
            },
            {
                "org": "NCHPAD",
                "title": "Upper Body Workout for Wheelchair Users | Seated Strength Training (video)",
                "url": "https://www.nchpad.org/resources/upper-body-workout-for-wheelchair-users-seated-strength-training/",
            },
        ],
    },
    "standing calf raise": {
        "match": ["calf raise", "heel raise", "calf", "toe stand"],
        "checkpoints": [
            "stand behind a sturdy chair or near a stable surface, holding on lightly for balance, feet shoulder-width apart",
            "rise smoothly up onto the balls of the feet, as high as comfortable",
            "pause briefly at the top, then lower the heels to the floor under control",
            "keep the ankles tracking straight, not rolling outward",
        ],
        "faults": ["bouncing", "rolling onto the outer edge of the foot", "short range of motion"],
        "adaptive": "Keep a hand on a wall or rail for balance; do it seated with light resistance if standing balance is limited.",
        "sources": [
            {
                "org": "NIA (NIH)",
                "title": "Workout to Go: Toe Stand",
                "url": "https://www.goaging.org/wp-content/uploads/2015/11/workout_to_go.pdf",
            },
        ],
    },
    "chair-assisted lunge": {
        "match": ["lunge", "split squat", "chair lunge", "static lunge"],
        "checkpoints": [
            "one hand on a sturdy chair for balance",
            "step one foot forward into a split stance",
            "drop the hips straight down rather than driving them forward, so the front knee stays roughly over the ankle",
            "keep the torso upright without swaying side to side; press back up through the front foot",
        ],
        "faults": ["front knee drifting far past the toes", "leaning or swaying the torso", "losing balance from too-narrow a stance"],
        "adaptive": "The chair gives balance support; reduce the range or hold the bottom briefly rather than full depth to start.",
        "sources": [
            {
                "org": "ACE",
                "title": "Exercise Library: Forward Lunge",
                "url": "https://www.acefitness.org/resources/everyone/exercise-library/94/forward-lunge/",
            },
        ],
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


def _format_sources(entry: dict) -> str:
    """One line naming the entry's specific references (org: title)."""
    return "; ".join(f"{s['org']}: {s['title']}" for s in entry["sources"])


def as_prompt_context(exercise_name: str) -> str:
    """A compact, injectable block of grounded standards for the coaching
    prompts. Empty string if the exercise isn't in the table (the model then
    falls back to its own knowledge, which is fine for out-of-table asks)."""
    entry = lookup(exercise_name)
    if not entry:
        return ""
    return (
        "\n\nCurated form standards for this exercise (base your coaching on these). "
        "They are plain-language summaries reconciled against the published references "
        "listed below. If you mention a source, name it accurately (e.g. 'guidance from "
        "the American Council on Exercise') and never present a cue as a verbatim quote:\n"
        f"- Correct checkpoints: {'; '.join(entry['checkpoints'])}.\n"
        f"- Common faults to watch for: {', '.join(entry['faults'])}.\n"
        f"- Adaptive modification: {entry['adaptive']}\n"
        f"- References: {_format_sources(entry)}."
    )


if __name__ == "__main__":
    for name in ["bodyweight squat", "wall push-up", "shoulder press", "jumping jacks"]:
        hit = lookup(name)
        print(f"{name!r} -> {_format_sources(hit) if hit else 'no match (falls back to model knowledge)'}")
