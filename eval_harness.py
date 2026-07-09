"""
eval_harness.py

Evaluation harness for the Audio Coach script pipeline. Runs a fixed set of
exercise prompts through generate_draft -> review_script (the default
local+Claude-QC path) or draft_with_claude (--claude-draft), then uses
Claude as an LLM judge to check each resulting script against the same
accessibility checklist the review prompt itself encodes.

This exists to catch regressions: if a future prompt tweak, model swap, or
"quick fix" quietly breaks accessibility compliance, this is what would
have caught it before a user did.

Usage:
    python eval_harness.py                 # run the full test set (local draft + Claude review)
    python eval_harness.py --limit 5       # only run the first 5 cases -- do this first
    python eval_harness.py --claude-draft  # use draft_with_claude instead of local+review

Requires: Ollama running with DEFAULT_MODEL pulled (unless --claude-draft),
          ANTHROPIC_API_KEY set (the judge always uses Claude, regardless
          of which path generated the script).
"""

import argparse
import json
import os
import re
import sys

import anthropic
from dotenv import load_dotenv

from ollama_draft import generate_draft, DEFAULT_MODEL
from claude_review import review_script, draft_with_claude
from model_manager import switch_to

load_dotenv()

# Deliberately varied: different joints/planes of motion, different contexts
# (seated, wheelchair, balance-sensitive, floor-based), since the checklist
# violations we've actually hit in practice (visual leakage, unpaired clock
# positions) showed up inconsistently across exercise types, not uniformly.
TEST_CASES = [
    {"exercise": "bodyweight squat", "context": "beginner, no equipment"},
    {"exercise": "wall push-up", "context": ""},
    {"exercise": "seated shoulder press", "context": "wheelchair user"},
    {"exercise": "standing calf raise", "context": ""},
    {"exercise": "chair-assisted lunge", "context": "beginner"},
    {"exercise": "standing arm circles", "context": ""},
    {"exercise": "torso rotation stretch", "context": ""},
    {"exercise": "diagonal wood chop", "context": "intermediate"},
    {"exercise": "glute bridge", "context": "lying on the floor"},
    {"exercise": "overhead arm raise", "context": "seated, beginner"},
    {"exercise": "standing hamstring stretch", "context": "balance concerns"},
    {"exercise": "bicep curl", "context": "using a resistance band"},
    {"exercise": "plank hold", "context": "beginner"},
    {"exercise": "step-up", "context": "using a low step or stair"},
    {"exercise": "side lateral raise", "context": ""},
]

JUDGE_SYSTEM_PROMPT = """You are auditing a verbal exercise script written for a blind or
low-vision listener. Respond with ONLY a JSON object, no other text, no markdown fences:

{
  "banned_visual_words": [list any of these found verbatim, case-insensitive, empty list if none:
    look, looks, looking, see, seeing, watch, watching, appears, visually, as shown, as pictured,
    as illustrated, over there, like this, like so],
  "unpaired_clock_positions": [list each sentence containing a clock position like "3 o'clock"
    that does NOT also contain a plain relative direction -- forward, back, left, right, up, down,
    side, toward, away, front, behind -- in that same sentence. Empty list if none, or if the
    script uses no clock positions at all.],
  "step_count": <integer: how many numbered steps are present>,
  "has_preamble_or_closing": <true if there is a title, introduction, or closing remark outside
    the numbered steps; false otherwise>,
  "pass": <true only if banned_visual_words is empty AND unpaired_clock_positions is empty AND
    step_count is exactly 10 AND has_preamble_or_closing is false>
}
"""


def judge_script(script: str, client: anthropic.Anthropic) -> dict:
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Script to audit:\n\n{script}"}],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N test cases.")
    parser.add_argument("--claude-draft", action="store_true", help="Use draft_with_claude instead of local draft + review.")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY is not set -- the judge needs it even when testing the local draft path.")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    cases = TEST_CASES[: args.limit] if args.limit else TEST_CASES
    results = []

    for i, case in enumerate(cases, 1):
        exercise, context = case["exercise"], case["context"]
        print(f"[{i}/{len(cases)}] {exercise} ({context or 'no context'})...", end=" ", flush=True)
        try:
            if args.claude_draft:
                script = draft_with_claude(exercise, extra_context=context)
            else:
                switch_to(DEFAULT_MODEL)
                draft = generate_draft(exercise, extra_context=context)
                script = review_script(draft, exercise_context=context)

            verdict = judge_script(script, client)
            results.append({"exercise": exercise, "context": context, "script": script, "verdict": verdict})
            print("PASS" if verdict["pass"] else "FAIL")
        except Exception as e:
            results.append({"exercise": exercise, "context": context, "error": str(e)})
            print(f"ERROR: {e}")

    passed = sum(1 for r in results if r.get("verdict", {}).get("pass"))
    failed = [r for r in results if not r.get("verdict", {}).get("pass", False)]

    print(f"\n{'=' * 60}")
    print(f"RESULT: {passed}/{len(results)} passed")
    print(f"{'=' * 60}")

    for r in failed:
        print(f"\n--- FAILED: {r['exercise']} ---")
        if "error" in r:
            print(f"  Error: {r['error']}")
            continue
        v = r["verdict"]
        if v.get("banned_visual_words"):
            print(f"  Banned visual words: {v['banned_visual_words']}")
        if v.get("unpaired_clock_positions"):
            print(f"  Unpaired clock positions: {v['unpaired_clock_positions']}")
        if v.get("step_count") != 10:
            print(f"  Step count: {v.get('step_count')} (expected 10)")
        if v.get("has_preamble_or_closing"):
            print("  Has preamble or closing remarks")
        print(f"  Full script:\n{r['script']}")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to eval_results.json")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
