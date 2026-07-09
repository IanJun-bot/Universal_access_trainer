"""
Unit tests for claude_review.py -- the retry-or-raise wrapper around Claude.

These recreate (as a permanent, committed test file) the checks used when the
empty-script bug was fixed: extended thinking occasionally consumed the whole
token budget and the API returned no usable text, which the app used to pass
through silently as an empty script. _complete_text must retry that case and
raise -- never return "" -- if every attempt fails.

No network calls: the Anthropic client is mocked throughout.

Run with:  python -m unittest test_claude_review -v
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import claude_review
from claude_review import MAX_RETRIES, _complete_text, _get_client


def _fake_response(text_blocks, stop_reason="end_turn"):
    """Build a stand-in for anthropic's Message: .content blocks + .stop_reason."""
    response = MagicMock()
    response.stop_reason = stop_reason
    content = []
    for text in text_blocks:
        block = MagicMock()
        block.type = "text"
        block.text = text
        content.append(block)
    response.content = content
    return response


def _thinking_only_response(stop_reason="end_turn"):
    """A response whose only block is thinking (no text) -- the bug's signature."""
    response = MagicMock()
    response.stop_reason = stop_reason
    block = MagicMock()
    block.type = "thinking"
    block.text = ""  # never read; type-gated out
    response.content = [block]
    return response


class TestCompleteText(unittest.TestCase):
    def test_returns_text_on_first_good_response(self):
        client = MagicMock()
        client.messages.create.return_value = _fake_response(["1. Stand tall."])
        result = _complete_text(client, model="m", system="s", user_content="u")
        self.assertEqual(result, "1. Stand tall.")
        self.assertEqual(client.messages.create.call_count, 1)

    def test_concatenates_multiple_text_blocks(self):
        client = MagicMock()
        client.messages.create.return_value = _fake_response(["1. First.", "2. Second."])
        result = _complete_text(client, model="m", system="s", user_content="u")
        self.assertEqual(result, "1. First.2. Second.")

    def test_retries_after_empty_response_then_succeeds(self):
        client = MagicMock()
        client.messages.create.side_effect = [
            _thinking_only_response(),               # attempt 1: no text block
            _fake_response(["1. Recovered."]),        # attempt 2: fine
        ]
        result = _complete_text(client, model="m", system="s", user_content="u")
        self.assertEqual(result, "1. Recovered.")
        self.assertEqual(client.messages.create.call_count, 2)

    def test_truncated_max_tokens_response_is_treated_as_failure(self):
        client = MagicMock()
        client.messages.create.side_effect = [
            _fake_response(["1. Cut off mid-"], stop_reason="max_tokens"),  # partial text: unusable
            _fake_response(["1. Complete script."]),
        ]
        result = _complete_text(client, model="m", system="s", user_content="u")
        self.assertEqual(result, "1. Complete script.")

    def test_raises_instead_of_returning_empty_after_all_retries(self):
        client = MagicMock()
        client.messages.create.return_value = _thinking_only_response(stop_reason="max_tokens")
        with self.assertRaises(RuntimeError) as ctx:
            _complete_text(client, model="m", system="s", user_content="u")
        # All attempts consumed, error names the last stop_reason for debugging.
        self.assertEqual(client.messages.create.call_count, MAX_RETRIES + 1)
        self.assertIn("max_tokens", str(ctx.exception))


class TestGetClient(unittest.TestCase):
    def test_missing_api_key_raises_clear_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                _get_client()
            self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))


class TestPublicPaths(unittest.TestCase):
    """review_script and draft_with_claude both route through _complete_text."""

    @patch("claude_review._get_client")
    def test_review_script_passes_draft_through(self, mock_get_client):
        client = MagicMock()
        client.messages.create.return_value = _fake_response(["1. Reviewed."])
        mock_get_client.return_value = client
        result = claude_review.review_script("1. Draft.", exercise_context="beginner")
        self.assertEqual(result, "1. Reviewed.")
        sent = client.messages.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("1. Draft.", sent)
        self.assertIn("beginner", sent)

    @patch("claude_review._get_client")
    def test_draft_with_claude_injects_form_standards(self, mock_get_client):
        client = MagicMock()
        client.messages.create.return_value = _fake_response(["1. Script."])
        mock_get_client.return_value = client
        result = claude_review.draft_with_claude("bodyweight squat")
        self.assertEqual(result, "1. Script.")
        sent = client.messages.create.call_args.kwargs["messages"][0]["content"]
        # The grounded standards block for squats must ride along in the prompt.
        self.assertIn("Curated form standards", sent)
        self.assertIn("knees track in line with the feet", sent)


if __name__ == "__main__":
    unittest.main()
