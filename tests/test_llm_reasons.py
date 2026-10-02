"""Unit tests for LLM error reasons, raw model output replays, and schema edge cases."""

from __future__ import annotations

import json
import pytest

from src.llm_client import LLMError, classify_llm_exception
from src.llm_tasks import parse_intent, ParseFailed
from src.graph import build_graph
from tests.test_graph_flow import FakeWeatherClient, FakeLLMClient


class MockException(Exception):
    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


def test_classify_llm_exception():
    # 429 Rate Limited
    assert classify_llm_exception(MockException("Resource exhausted", code=429)) == "rate_limited"
    assert classify_llm_exception(Exception("429 RESOURCE_EXHAUSTED")) == "rate_limited"
    assert classify_llm_exception(Exception("rate limit exceeded")) == "rate_limited"

    # Timeout
    assert classify_llm_exception(TimeoutError("Request timed out")) == "api_error:timeout"
    assert classify_llm_exception(Exception("Client timeout occurred")) == "api_error:timeout"

    # Safety
    assert classify_llm_exception(Exception("Response blocked by safety policy")) == "safety_blocked"

    # Status codes
    assert classify_llm_exception(MockException("Unavailable", code=503)) == "api_error:503"
    assert classify_llm_exception(Exception("503 UNAVAILABLE. High demand")) == "api_error:503"
    assert classify_llm_exception(Exception("500 Internal Server Error")) == "api_error:500"


def test_replay_api_error_503():
    """Replay the root cause: transient 503 UNAVAILABLE from Gemini."""
    err = LLMError("503 UNAVAILABLE: High demand spike", reason="api_error:503")
    client = FakeLLMClient(parse_responses=[err, err])

    vocab = {"cycling": {"title": "Cycling", "intent_description": "Rules for cycling."}}
    with pytest.raises(ParseFailed) as exc_info:
        parse_intent(client, "is it safe to cycle in Bhopal today?", vocab)

    assert exc_info.value.reason == "api_error:503"
    assert "api_error:503" in str(exc_info.value)
    # Ensure no secrets or user text in reason
    assert "is it safe to cycle" not in exc_info.value.reason


def test_replay_rate_limited_429():
    """Replay 429 RESOURCE_EXHAUSTED."""
    err = LLMError("429 RESOURCE_EXHAUSTED", reason="rate_limited")
    client = FakeLLMClient(parse_responses=[err, err])

    vocab = {"cycling": {"title": "Cycling", "intent_description": "Rules for cycling."}}
    with pytest.raises(ParseFailed) as exc_info:
        parse_intent(client, "is it safe to cycle in Bhopal today?", vocab)

    assert exc_info.value.reason == "rate_limited"


def test_replay_invalid_json():
    """Replay model returning broken, unparseable JSON."""
    bad_json = """```json
{
  "intent": "advice",
  "location": "Bhopal",
  "activity_tags": ["cycling"
```"""
    client = FakeLLMClient(parse_responses=[bad_json, bad_json])

    vocab = {"cycling": {"title": "Cycling", "intent_description": "Rules for cycling."}}
    with pytest.raises(ParseFailed) as exc_info:
        parse_intent(client, "is it safe to cycle in Bhopal today?", vocab)

    assert exc_info.value.reason == "invalid_json"


def test_replay_schema_violation_intent():
    """Replay model returning invalid intent enum."""
    bad_schema = json.dumps({
        "intent": "unsupported_action",
        "location": "Bhopal",
        "activity_tags": ["cycling"],
        "time_ref": "today",
        "cited_sop_ids": [],
    })
    client = FakeLLMClient(parse_responses=[bad_schema, bad_schema])

    vocab = {"cycling": {"title": "Cycling", "intent_description": "Rules for cycling."}}
    with pytest.raises(ParseFailed) as exc_info:
        parse_intent(client, "is it safe to cycle in Bhopal today?", vocab)

    assert exc_info.value.reason == "schema_violation:intent"


def test_retry_recovers_after_transient_spike():
    """Verify GeminiClient retries transparently on 503 and succeeds on second attempt."""
    from unittest.mock import MagicMock
    from src.llm_client import GeminiClient

    client = GeminiClient(
        api_key="fake-test-key",
        model_name="gemini-test",
        backoffs=[0.0, 0.0],
    )

    attempts = 0
    valid_output = json.dumps({
        "intent": "advice",
        "location": "Bhopal",
        "activity_tags": ["cycling"],
        "time_ref": "today",
        "cited_sop_ids": [],
    })

    def mock_generate_content(model, contents, config):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise Exception("503 UNAVAILABLE. High demand spike")
        mock_resp = MagicMock()
        mock_resp.text = valid_output
        return mock_resp

    client._client.models.generate_content = mock_generate_content

    vocab = {"cycling": {"title": "Cycling", "intent_description": "Rules for cycling."}}
    parsed = parse_intent(client, "is it safe to cycle in Bhopal today?", vocab)
    assert parsed.intent == "advice"
    assert parsed.location == "Bhopal"
    assert parsed.activity_tags == ["cycling"]
    assert attempts == 2


def test_robust_json_parsing_and_null_coercion():
    """Replay model output wrapped in conversational text with null fields."""
    raw_output = """Here is the extracted information in JSON format:
```json
{
  "intent": "advice",
  "location": "Bhopal",
  "activity_tags": null,
  "time_ref": "",
  "cited_sop_ids": null
}
```
Let me know if you need more details!"""

    client = FakeLLMClient(parse_responses=[raw_output])
    vocab = {"cycling": {"title": "Cycling", "intent_description": "Rules for cycling."}}
    parsed = parse_intent(client, "is it safe to cycle in Bhopal today?", vocab)

    assert parsed.intent == "advice"
    assert parsed.location == "Bhopal"
    assert parsed.activity_tags == []
    assert parsed.time_ref is None
    assert parsed.cited_sop_ids == []


def test_graph_flow_on_parse_failure():
    """Verify graph captures error_reason and routes to llm_unavailable on 503 API error."""
    err = LLMError("503 UNAVAILABLE", reason="api_error:503")
    failing_llm = FakeLLMClient(parse_responses=[err, err])

    graph = build_graph(
        weather_client=FakeWeatherClient(),
        llm=failing_llm,
    )

    config = {"configurable": {"thread_id": "test-parse-fail-1"}}
    result = graph.invoke({"query": "is it safe to cycle in Bhopal today?"}, config=config)

    assert result.get("kind") == "llm_unavailable"
    assert "temporarily unavailable" in result.get("reply", "")
    assert result.get("error_reason") == "api_error:503"

    decision_log = result.get("decision_log", [])
    assert len(decision_log) == 1
    assert decision_log[0]["kind"] == "llm_unavailable"
    assert decision_log[0]["error_reason"] == "api_error:503"


def test_graph_flow_on_invalid_json():
    """Verify graph routes to parse_failed on genuinely malformed JSON."""
    bad_json = "NOT_JSON_AT_ALL"
    failing_llm = FakeLLMClient(parse_responses=[bad_json, bad_json])

    graph = build_graph(
        weather_client=FakeWeatherClient(),
        llm=failing_llm,
    )

    config = {"configurable": {"thread_id": "test-invalid-json-1"}}
    result = graph.invoke({"query": "is it safe to cycle in Bhopal today?"}, config=config)

    assert result.get("kind") == "parse_failed"
    assert result.get("reply") == "I couldn't understand your request, please rephrase."
    assert result.get("error_reason") == "invalid_json"
