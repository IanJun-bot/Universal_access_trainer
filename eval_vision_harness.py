"""
eval_vision_harness.py

Evaluation harness for the Form Checker vision pipeline. Unlike
eval_harness.py, this cannot use synthetic test cases -- it needs real
photos of real people to mean anything, especially the deliberately
ambiguous ones (blurry, cropped, wrong angle) that test whether the model
actually says "can't assess form" instead of inventing confident-sounding
corrections for a photo it can't actually read. That confidence behavior
was added to the prompt but has never been checked against a real photo --
this harness is what closes that gap.

Setup:
    Create a folder called eval_photos/ next to this file and drop in real
    test photos (yourself, a friend, whoever will let you). Name files so
    the filename hints at what the photo is testing -- if a filename
    contains any of AMBIGUOUS_HINTS below, this harness expects a
    "can't assess" response and flags it if that doesn't happen.
    Everything else is treated as a normal case, printed for you to read
    and judge yourself -- there's no automated ground truth for "is this
    correction actually good advice," only for the confidence behavior.

    Example filenames:
        squat_good_angle.jpg
        pushup_side_view.jpg
        squat_blurry.jpg          <- expected to trigger "can't assess"
        pushup_cropped_feet.jpg   <- expected to trigger "can't assess"
        lunge_dark_room.jpg       <- expected to trigger "can't assess"

Usage:
    python eval_vision_harness.py           # local Ollama vision model
    python eval_vision_harness.py --claude  # Claude vision API instead
"""

import argparse
import sys
from pathlib import Path

from vision_form_checker import check_form
from claude_vision import check_form_with_claude

PHOTOS_DIR = Path(__file__).parent / "eval_photos"
AMBIGUOUS_HINTS = ["blurry", "blur", "cropped", "crop", "dark", "bad_angle", "far", "partial"]
CANT_ASSESS_PHRASES = ["can't assess", "cannot assess", "can not assess", "doesn't show", "does not show"]
EXERCISE_HINTS = ["squat", "push-up", "pushup", "lunge", "press", "raise", "plank", "curl"]


def guess_exercise_name(stem: str) -> str:
    stem_lower = stem.lower()
    for word in EXERCISE_HINTS:
        if word in stem_lower:
            return word.replace("-", " ")
    return "exercise"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--claude", action="store_true", help="Use the Claude vision path instead of the local model.")
    args = parser.parse_args()

    if not PHOTOS_DIR.exists():
        print(f"No {PHOTOS_DIR} folder found.")
        print("Create it and add real test photos -- see this file's docstring for naming conventions.")
        sys.exit(1)

    photos = sorted(
        p for p in PHOTOS_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not photos:
        print(f"{PHOTOS_DIR} exists but has no .jpg/.jpeg/.png files in it yet.")
        sys.exit(1)

    print(f"Running {len(photos)} photo(s) through the {'Claude' if args.claude else 'local'} vision path...\n")

    flagged = []
    for photo_path in photos:
        exercise = guess_exercise_name(photo_path.stem)
        should_be_ambiguous = any(hint in photo_path.stem.lower() for hint in AMBIGUOUS_HINTS)

        image_bytes = photo_path.read_bytes()
        try:
            captions = check_form_with_claude(image_bytes, exercise) if args.claude else check_form(image_bytes, exercise)
        except Exception as e:
            print(f"--- {photo_path.name} ---\n[ERROR] {e}\n")
            continue

        said_cant_assess = any(phrase in captions.lower() for phrase in CANT_ASSESS_PHRASES)

        if should_be_ambiguous and not said_cant_assess:
            status = "FLAG: expected \"can't assess\", got confident corrections instead"
            flagged.append(photo_path.name)
        elif not should_be_ambiguous and said_cant_assess:
            status = "NOTE: said \"can't assess\" on a photo not marked ambiguous -- check if it actually is"
        else:
            status = "OK"

        print(f"--- {photo_path.name} ({'expected ambiguous' if should_be_ambiguous else 'expected assessable'}) ---")
        print(captions)
        print(f"[{status}]\n")

    print("=" * 60)
    if flagged:
        print(f"{len(flagged)} photo(s) FAILED the confidence check: {flagged}")
        sys.exit(1)

    print("All photos handled the confidence check as expected.")
    print("This only checks whether the model correctly abstains on ambiguous photos --")
    print("you still need to read every caption above yourself to judge whether the")
    print("actual corrections given on the normal photos are good advice.")
    sys.exit(0)


if __name__ == "__main__":
    main()
