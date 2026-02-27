"""
tests/test_lm_studio.py
Tests for LM Studio connectivity and prompt builder.
LM Studio connectivity tests are skipped if LM Studio is not running.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

SRC = Path(__file__).parent.parent
sys.path.insert(0, str(SRC))

from app import build_prompt, call_lm_studio, extract_next_action

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"


def lm_studio_available() -> bool:
    try:
        requests.get("http://localhost:1234/v1/models", timeout=2)
        return True
    except Exception:
        return False


class TestExtractNextAction:
    def test_extracts_standard_line(self):
        analysis = "NEXT ACTION: Apologize and offer a credit.\n\n**1. Next Event**\nHold"
        assert extract_next_action(analysis) == "Apologize and offer a credit."

    def test_case_insensitive(self):
        analysis = "next action: Ask the customer for their account number."
        assert extract_next_action(analysis) == "Ask the customer for their account number."

    def test_returns_none_when_missing(self):
        analysis = "**1. Next Event**\nHold\n**2. Sentiment**\nNegative"
        assert extract_next_action(analysis) is None

    def test_returns_none_for_empty_action(self):
        analysis = "NEXT ACTION:\n**1. Next Event**\nHold"
        assert extract_next_action(analysis) is None

    def test_handles_leading_whitespace(self):
        analysis = "  NEXT ACTION: Transfer the call to billing.\n**Details**"
        assert extract_next_action(analysis) == "Transfer the call to billing."

    def test_ignores_next_action_mid_text(self):
        # Only the first matching line should be used
        analysis = "NEXT ACTION: Say hello.\nSome other next action: something else."
        assert extract_next_action(analysis) == "Say hello."


class TestBuildPrompt:
    def _make_retrieved(self, n: int = 3) -> list[dict]:
        return [
            {
                "call_id": f"CALL-{i:08d}",
                "category": "Billing",
                "sub_category": "High charges",
                "similarity": 0.9 - i * 0.05,
                "transcript_text": f"Agent: Hello\nCustomer: My bill is high #{i}",
            }
            for i in range(n)
        ]

    def test_returns_two_messages(self):
        msgs = build_prompt("Agent: Hi\nCustomer: Help", self._make_retrieved())
        assert len(msgs) == 2

    def test_system_role_present(self):
        msgs = build_prompt("Agent: Hi\nCustomer: Help", self._make_retrieved())
        assert msgs[0]["role"] == "system"

    def test_user_role_present(self):
        msgs = build_prompt("Agent: Hi\nCustomer: Help", self._make_retrieved())
        assert msgs[1]["role"] == "user"

    def test_current_transcript_in_prompt(self):
        transcript = "Agent: Hi there\nCustomer: I have a billing problem"
        msgs = build_prompt(transcript, self._make_retrieved())
        assert transcript in msgs[1]["content"]

    def test_retrieved_context_in_prompt(self):
        retrieved = self._make_retrieved(2)
        msgs = build_prompt("Agent: Hi", retrieved)
        content = msgs[1]["content"]
        assert "Billing" in content
        assert "Similar Call 1" in content

    def test_empty_retrieved_still_builds(self):
        msgs = build_prompt("Agent: Hi", [])
        assert len(msgs) == 2
        assert "CURRENT CONVERSATION" in msgs[1]["content"]


class TestCallLMStudio:
    def test_connection_error_returns_friendly_message(self):
        with patch("app.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError()
            result = call_lm_studio([{"role": "user", "content": "test"}])
        assert "Error" in result
        assert "LM Studio" in result

    def test_successful_response_extracted(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "  Great analysis!  "}}]
        }
        mock_response.raise_for_status.return_value = None
        with patch("app.requests.post", return_value=mock_response):
            result = call_lm_studio([{"role": "user", "content": "test"}])
        assert result == "Great analysis!"

    @pytest.mark.skipif(
        not lm_studio_available(),
        reason="LM Studio not running on localhost:1234",
    )
    def test_live_lm_studio_response(self):
        msgs = [
            {
                "role": "user",
                "content": (
                    "In one sentence, what is the capital of France? "
                    "Respond with just the answer."
                ),
            }
        ]
        result = call_lm_studio(msgs, temperature=0.0)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Paris" in result

    @pytest.mark.skipif(
        not lm_studio_available(),
        reason="LM Studio not running on localhost:1234",
    )
    def test_live_rag_analysis(self):
        transcript = (
            "Agent: Thank you for calling, how can I help you today?\n"
            "Customer: My bill is $200 higher than usual and I want answers NOW!\n"
            "Agent: I understand your frustration. Let me pull up your account."
        )
        retrieved = [
            {
                "call_id": "CALL-TEST001",
                "category": "Billing",
                "sub_category": "High charges dispute",
                "similarity": 0.88,
                "transcript_text": (
                    "Agent: How can I help?\n"
                    "Customer: My bill is wrong.\n"
                    "Agent: I'm sorry, let me check that for you.\n"
                    "Customer: It's $150 more than last month!\n"
                    "Agent: I can see there was a data overage. I'll apply a one-time credit."
                ),
            }
        ]
        msgs = build_prompt(transcript, retrieved)
        result = call_lm_studio(msgs, temperature=0.3)
        assert len(result) > 50
        # Should contain at least one of the expected sections
        keywords = ["sentiment", "Sentiment", "agent", "Agent", "script", "Script", "event", "Event"]
        assert any(k in result for k in keywords)
