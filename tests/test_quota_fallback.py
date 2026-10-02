"""Tests for quota handling, model fallbacks, cooldowns, and degradation outcomes.

All tests run strictly offline without network access using mocks and FakeLLMClient.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock
import pytest

from src.llm_client import (
    GeminiClient,
    DailyQuotaCooldownTracker,
    parse_fallback_models,
    LLMError,
)
from src.llm_tasks import parse_intent, ParseFailed
from src.graph import build_graph
from tests.test_graph_flow import FakeWeatherClient, FakeLLMClient


def test_fallback_list_parsing():
    """Verify fallback models parsing trims spaces, drops empties, and excludes primary duplicates."""
    primary = "gemini-2.5-flash"
    fallback_str = " gemini-2.0-flash ,  gemini-2.5-flash, gemini-1.5-flash, , gemini-2.0-flash "
    parsed = parse_fallback_models(primary, fallback_str)

    assert parsed == ["gemini-2.0-flash", "gemini-1.5-flash"]
    assert parse_fallback_models(primary, None) == []
    assert parse_fallback_models(primary, "") == []
    assert parse_fallback_models(primary, "   , ,  ") == []
    assert parse_fallback_models(primary, "gemini-2.5-flash") == []


def test_daily_quota_cooldown_tracker():
    """Verify daily-quota cooldown helper detects 'PerDay' errors and tracks models."""
    tracker = DailyQuotaCooldownTracker()
    model = "gemini-2.5-flash"

    assert not tracker.is_on_cooldown(model)

    err_daily = Exception("429 Resource has been exhausted (e.g. check quota): GenerateContentRequestsPerDay")
    err_minute = Exception("429 Resource has been exhausted (e.g. check quota): GenerateContentRequestsPerMinute")

    assert tracker.is_daily_quota_error(err_daily)
    assert not tracker.is_daily_quota_error(err_minute)

    tracker.mark_cooldown(model)
    assert tracker.is_on_cooldown(model)

    tracker.clear()
    assert not tracker.is_on_cooldown(model)


def test_primary_daily_quota_falls_back_to_next_model():
    """Primary returns daily-quota 429 -> fallback answers, model_used is fallback."""
    tracker = DailyQuotaCooldownTracker()
    client = GeminiClient(
        api_key="fake-test-key",
        model_name="gemini-primary",
        fallback_models=["gemini-fallback"],
        cooldown_tracker=tracker,
        backoffs=[0.0, 0.0],
    )

    calls = []

    def mock_generate_content(model, contents, config):
        calls.append(model)
        if model == "gemini-primary":
            raise Exception("429 Quota exceeded: GenerateContentRequestsPerDay")
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "intent": "advice",
            "location": "Bhopal",
            "activity_tags": ["cycling"],
            "time_ref": "today",
            "cited_sop_ids": [],
        })
        return mock_resp

    client._client.models.generate_content = mock_generate_content

    result = client.parse_intent_raw("system", "is it safe to cycle?")
    assert "cycling" in result
    assert client.model_used == "gemini-fallback"
    assert calls == ["gemini-primary", "gemini-fallback"]
    assert tracker.is_on_cooldown("gemini-primary")


def test_primary_per_minute_429_retries_and_succeeds():
    """Primary returns per-minute 429 twice then succeeds -> primary answers."""
    tracker = DailyQuotaCooldownTracker()
    client = GeminiClient(
        api_key="fake-test-key",
        model_name="gemini-primary",
        fallback_models=["gemini-fallback"],
        cooldown_tracker=tracker,
        backoffs=[0.0, 0.0],
    )

    attempts = 0

    def mock_generate_content(model, contents, config):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise Exception("429 RESOURCE_EXHAUSTED per minute")
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({"intent": "advice", "location": "Bhopal"})
        return mock_resp

    client._client.models.generate_content = mock_generate_content

    result = client.parse_intent_raw("system", "query")
    assert "advice" in result
    assert client.model_used == "gemini-primary"
    assert attempts == 3
    assert not tracker.is_on_cooldown("gemini-primary")


def test_cooldown_skips_exhausted_model_with_zero_attempts():
    """Exhausted model on cooldown gets zero attempts on the next turn."""
    tracker = DailyQuotaCooldownTracker()
    tracker.mark_cooldown("gemini-primary")

    client = GeminiClient(
        api_key="fake-test-key",
        model_name="gemini-primary",
        fallback_models=["gemini-fallback"],
        cooldown_tracker=tracker,
        backoffs=[0.0, 0.0],
    )

    calls = []

    def mock_generate_content(model, contents, config):
        calls.append(model)
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({"intent": "advice"})
        return mock_resp

    client._client.models.generate_content = mock_generate_content

    client.parse_intent_raw("system", "query")
    assert "gemini-primary" not in calls
    assert calls == ["gemini-fallback"]
    assert client.model_used == "gemini-fallback"


def test_404_on_primary_advances_immediately_without_retries():
    """404 on primary means model unavailable: skip that model with 0 retries."""
    tracker = DailyQuotaCooldownTracker()
    client = GeminiClient(
        api_key="fake-test-key",
        model_name="gemini-primary",
        fallback_models=["gemini-fallback"],
        cooldown_tracker=tracker,
        backoffs=[0.0, 0.0],
    )

    calls = []

    def mock_generate_content(model, contents, config):
        calls.append(model)
        if model == "gemini-primary":
            raise Exception("404 NOT_FOUND. Model does not exist")
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({"intent": "advice"})
        return mock_resp

    client._client.models.generate_content = mock_generate_content

    client.parse_intent_raw("system", "query")
    assert calls == ["gemini-primary", "gemini-fallback"]
    assert client.model_used == "gemini-fallback"


def test_every_model_fails_routes_to_llm_unavailable():
    """When all models fail, graph routes to kind llm_unavailable with no weather numbers."""
    failing_llm = FakeLLMClient(
        parse_responses=[LLMError("rate_limited", reason="rate_limited")],
        model_used=None,
    )

    graph = build_graph(
        weather_client=FakeWeatherClient(),
        llm=failing_llm,
    )

    config = {"configurable": {"thread_id": "test-exhausted-1"}}
    result = graph.invoke({"query": "is it safe to cycle in Bhopal today?"}, config=config)

    assert result.get("kind") == "llm_unavailable"
    reply = result.get("reply", "")
    assert "temporarily unavailable (usage limit reached)" in reply
    # Must give no weather numbers, forecasts or advice
    assert "30." not in reply
    assert "km/h" not in reply
    assert "°C" not in reply
    assert result.get("model_used") is None


def test_invalid_json_routes_to_parse_failed_no_model_switching():
    """Invalid JSON routes to kind parse_failed with distinct message and no model switching."""
    bad_json_llm = FakeLLMClient(
        parse_responses=["NOT_JSON"],
        model_used="gemini-primary",
    )

    graph = build_graph(
        weather_client=FakeWeatherClient(),
        llm=bad_json_llm,
    )

    config = {"configurable": {"thread_id": "test-bad-json-1"}}
    result = graph.invoke({"query": "is it safe to cycle in Bhopal today?"}, config=config)

    assert result.get("kind") == "parse_failed"
    assert result.get("reply") == "I couldn't understand your request, please rephrase."
    assert result.get("error_reason") == "invalid_json"


def test_safety_block_routes_to_request_blocked():
    """Safety-blocked response routes to kind request_blocked with honest message."""
    blocked_llm = FakeLLMClient(
        parse_responses=[LLMError("Safety blocked", reason="safety_blocked")],
        model_used=None,
    )

    graph = build_graph(
        weather_client=FakeWeatherClient(),
        llm=blocked_llm,
    )

    config = {"configurable": {"thread_id": "test-blocked-1"}}
    result = graph.invoke({"query": "dangerous query"}, config=config)

    assert result.get("kind") == "request_blocked"
    assert result.get("reply") == "I couldn't process that request."
    assert result.get("error_reason") == "safety_blocked"


def test_compose_failure_falls_back_to_templated_answer():
    """If compose fails (e.g. rate limit), keep existing templated_answer fallback with policy footer."""
    compose_failing_llm = FakeLLMClient(
        parse_responses=[json.dumps({
            "intent": "advice",
            "location": "Bhopal",
            "activity_tags": ["cycling"],
            "time_ref": "today",
            "cited_sop_ids": [],
        })],
        compose_responses=[LLMError("Compose rate limited", reason="rate_limited")],
        model_used="gemini-primary",
    )

    graph = build_graph(
        weather_client=FakeWeatherClient(),
        llm=compose_failing_llm,
    )

    config = {"configurable": {"thread_id": "test-compose-fail-1"}}
    result = graph.invoke({"query": "is it safe to cycle in Bhopal today?"}, config=config)

    assert result.get("kind") == "advice"
    reply = result.get("reply", "")
    # Check that templated answer worked and includes citation footer
    assert "Policy: CLR-EXR" in reply
    assert "Location: Bhopal, India" in reply
    assert "Data fetched:" in reply
