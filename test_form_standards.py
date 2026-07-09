"""
Unit tests for form_standards.py -- the curated, cited form-standards lookup.

Run with:  python -m unittest test_form_standards -v
"""

import unittest

from form_standards import FORM_STANDARDS, GENERAL_SOURCES, lookup, as_prompt_context


class TestLookup(unittest.TestCase):
    def test_matches_each_entry_by_common_names(self):
        cases = {
            "bodyweight squat": "squat",
            "sit to stand": "squat",
            "wall push-up": "wall push-up",
            "incline pushup": "wall push-up",
            "seated shoulder press": "seated shoulder press",
            "overhead press with dumbbells": "seated shoulder press",
            "standing calf raise": "standing calf raise",
            "toe stand": "standing calf raise",
            "chair-assisted lunge": "chair-assisted lunge",
            "split squat": "squat",  # "squat" match-term wins (dict order); still a valid entry
        }
        for query, expected_key in cases.items():
            with self.subTest(query=query):
                entry = lookup(query)
                self.assertIsNotNone(entry, f"no match for {query!r}")
                self.assertEqual(entry, FORM_STANDARDS[expected_key])

    def test_case_insensitive(self):
        self.assertIsNotNone(lookup("Bodyweight SQUAT"))

    def test_no_match_returns_none(self):
        self.assertIsNone(lookup("jumping jacks"))
        self.assertIsNone(lookup(""))
        self.assertIsNone(lookup(None))


class TestEntryShape(unittest.TestCase):
    """Every entry must carry complete, well-formed content and citations."""

    def test_entries_are_complete(self):
        for name, entry in FORM_STANDARDS.items():
            with self.subTest(entry=name):
                self.assertTrue(entry["match"], "match terms missing")
                self.assertGreaterEqual(len(entry["checkpoints"]), 3)
                self.assertTrue(entry["faults"])
                self.assertTrue(entry["adaptive"].strip())
                self.assertTrue(entry["sources"], "entry has no citations")

    def test_sources_are_well_formed(self):
        all_sources = list(GENERAL_SOURCES)
        for entry in FORM_STANDARDS.values():
            all_sources.extend(entry["sources"])
        for src in all_sources:
            with self.subTest(source=src.get("title", "?")):
                self.assertTrue(src["org"].strip())
                self.assertTrue(src["title"].strip())
                self.assertTrue(src["url"].startswith("https://"), f"non-https url: {src['url']}")


class TestPromptContext(unittest.TestCase):
    def test_unknown_exercise_returns_empty(self):
        self.assertEqual(as_prompt_context("jumping jacks"), "")
        self.assertEqual(as_prompt_context(""), "")

    def test_known_exercise_includes_content_and_references(self):
        block = as_prompt_context("bodyweight squat")
        entry = FORM_STANDARDS["squat"]
        for checkpoint in entry["checkpoints"]:
            self.assertIn(checkpoint, block)
        self.assertIn(entry["adaptive"], block)
        for src in entry["sources"]:
            self.assertIn(src["title"], block)

    def test_block_instructs_accurate_attribution(self):
        # The injected block must forbid presenting cues as verbatim quotes.
        block = as_prompt_context("wall push-up")
        self.assertIn("never present a cue as a verbatim quote", block)


if __name__ == "__main__":
    unittest.main()
