"""Tests for Pre-Step 6 enhancements:
1. Compose policy: clear or info primary SOP skips LLM (answer_source="template", reason="no_llm_needed").
2. Footer: deterministic 'Conditions' line built by code with units and window bounds.
3. Window note: appears exactly once, added by code, compose prompt does not mention window, 'midnight' used.
4. 'Data fetched' formatted as local time with timezone abbreviation, no microseconds.
5. Strict zero-network / fake LLM testing.
"""

from __future__ import annotations

import datetime
import json
from langgraph.checkpoint.memory import MemorySaver

from src.graph import build_graph
from src.llm_tasks import compose_answer
from src.sop_engine import SOPResult
from src.validator import (
    build_footer,
    format_conditions_line,
    format_local_fetch_time,
    templated_answer,
)
from src.weather_client import (
    RawWeatherData,
    OpenMeteoClient,
)
from tests.test_graph_flow import FakeLLMClient, FakeWeatherClient


def test_clear_sop_skips_llm_compose():
    """Verify that match_type clear SOP skips LLM compose call and sets template/no_llm_needed."""
    llm = FakeLLMClient(
        parse_responses=[
            json.dumps({"intent": "advice", "location": "Bhopal", "activity_tags": ["cycling"], "time_ref": "today"})
        ]
    )
    # Default facts trigger CLR-EXR (clear SOP)
    weather_client = FakeWeatherClient()
    graph = build_graph(weather_client=weather_client, llm=llm, checkpointer=MemorySaver())

    result = graph.invoke(
        {"query": "Is it safe to cycle in Bhopal today?"},
        config={"configurable": {"thread_id": "test-clear-skip-llm"}},
    )

    # LLM compose must NEVER be called for clear SOP
    assert len(llm.recorded_compose_payloads) == 0
    assert result["answer_source"] == "template"
    assert result["reason"] == "no_llm_needed"
    assert "Policy: CLR-EXR (info)" in result["reply"]


def test_info_severity_hazard_skips_llm_compose():
    """Verify that an SOP with effective severity info skips LLM compose."""
    from src.nodes import GraphNodes
    from src.facts_registry import FactsRegistry
    from src.sop_loader import load_sops

    sops = load_sops("sops")
    facts_reg = FactsRegistry()
    fake_llm = FakeLLMClient()
    nodes = GraphNodes(
        weather_client=FakeWeatherClient(),
        llm=fake_llm,
        sops=sops,
        facts_registry=facts_reg,
        vocabulary={},
    )

    # Primary with effective_severity info
    info_primary = SOPResult(
        sop_id="INFO-001",
        match_type="numeric",
        effective_severity="info",
        priority=10,
        rendered_advice="Wear sunglasses if needed.",
        facts_used={"uv_index_max": 3.0},
        status="matched",
    )

    state = {
        "primary": info_primary.model_dump(),
        "also_applies": [],
        "activity_tags": ["cycling"],
        "resolved_name": "Bhopal, India",
        "time_ref": "today",
    }

    compose_res = nodes.compose_node(state)
    assert compose_res["_raw_composed"] is None
    assert compose_res["answer_source"] == "template"
    assert compose_res["reason"] == "no_llm_needed"
    assert len(fake_llm.recorded_compose_payloads) == 0


def test_hazard_sop_uses_llm_compose():
    """Verify that hazard SOPs with low-or-above severity invoke the LLM composer."""
    llm = FakeLLMClient(
        parse_responses=[
            json.dumps({"intent": "advice", "location": "Bhopal", "activity_tags": ["cycling"], "time_ref": "today"})
        ],
        compose_responses=["High wind alert in Bhopal! Wind gusts reach 42.0 km/h. Avoid cycling."],
    )
    # Wind gusts 42.0 triggers EXR-001 (high severity)
    weather_client = FakeWeatherClient(facts_override={"wind_gusts": 42.0})
    graph = build_graph(weather_client=weather_client, llm=llm, checkpointer=MemorySaver())

    result = graph.invoke(
        {"query": "Is it safe to cycle in Bhopal today?"},
        config={"configurable": {"thread_id": "test-hazard-calls-llm"}},
    )

    # LLM compose was invoked
    assert len(llm.recorded_compose_payloads) == 1
    assert result["answer_source"] == "llm"
    assert result["reason"] == "llm_composed"
    assert "High wind alert in Bhopal!" in result["reply"]
    assert "Policy: EXR-" in result["reply"]
    assert "(high)" in result["reply"]


def test_conditions_line_formatting_with_facts_used():
    """Test format_conditions_line with specific facts used and units."""
    facts_used = {
        "wind_gusts": 18.0,
        "precipitation_sum": 0.0,
        "uv_index_max": 4.5,
    }
    line = format_conditions_line("17:00-midnight", facts_used=facts_used)
    assert line == "Conditions (17:00-midnight): max wind gusts 18.0 km/h, rain 0.0 mm, max UV 4.5"


def test_conditions_line_fallback_to_key_facts_when_facts_used_empty():
    """Test format_conditions_line falls back to key facts for clear SOPs."""
    all_facts = {
        "wind_gusts": 22.4,
        "precipitation_sum": 1.2,
        "uv_index_max": 5.0,
        "temperature": 28.0,
    }
    line = format_conditions_line("today", facts_used={}, all_facts=all_facts)
    assert line is not None
    assert "Conditions (today):" in line
    assert "max wind gusts 22.4 km/h" in line
    assert "rain 1.2 mm" in line
    assert "max UV 5.0" in line


def test_footer_contains_conditions_line_and_local_fetch_time():
    """Verify build_footer formats Conditions line and local fetch time with timezone abbreviation."""
    primary = SOPResult(
        sop_id="EXR-001",
        match_type="numeric",
        effective_severity="high",
        priority=90,
        rendered_advice="Severe wind warning.",
        facts_used={"wind_gusts": 42.0},
        status="matched",
    )
    raw_fetch = "2026-10-02T10:00:00.123456Z"
    footer = build_footer(
        primary=primary,
        also_applies=[],
        resolved_name="Bhopal, Madhya Pradesh, India",
        fetch_time=raw_fetch,
        conditions_line="Conditions (17:00-midnight): max wind gusts 42.0 km/h",
        timezone_name="Asia/Kolkata",
        utc_offset_seconds=19800,
    )

    assert "Policy: EXR-001 (high)" in footer
    assert "Conditions (17:00-midnight): max wind gusts 42.0 km/h" in footer
    assert "Location: Bhopal, Madhya Pradesh, India" in footer
    # Asia/Kolkata is +05:30 -> 10:00 UTC becomes 15:30 IST
    assert "Data fetched: 2026-10-02 15:30 IST" in footer
    assert ".123456" not in footer  # No microseconds

    # Verify each footer item is on its own line/block separated by "\n\n"
    parts = footer.strip().split("\n\n")
    assert parts[0] == "---"
    assert parts[1] == "Policy: EXR-001 (high)"
    assert parts[2] == "Conditions (17:00-midnight): max wind gusts 42.0 km/h"
    assert parts[3] == "Location: Bhopal, Madhya Pradesh, India"
    assert parts[4] == "Data fetched: 2026-10-02 15:30 IST"


def test_format_local_fetch_time_variations():
    """Test format_local_fetch_time with different timezone offsets and inputs."""
    # UTC ISO string with microseconds -> Asia/Kolkata
    t1 = format_local_fetch_time("2026-10-02T11:08:43.655476+00:00", timezone_name="Asia/Kolkata")
    assert t1 == "2026-10-02 16:38 IST"

    # UTC string with Z
    t2 = format_local_fetch_time("2026-10-02T05:00:00Z", timezone_name="UTC")
    assert t2 == "2026-10-02 05:00 UTC"

    # Offset only fallback
    t3 = format_local_fetch_time("2026-10-02T12:00:00Z", timezone_name=None, utc_offset_seconds=3600)
    assert "2026-10-02 13:00" in t3
    assert "." not in t3


def test_window_note_appears_exactly_once_in_template_path():
    """Verify window_note appears exactly once in templated responses."""
    text = templated_answer(
        primary=None,
        also_applies=[],
        resolved_name="Bhopal",
        time_ref="this_evening",
        fetch_time="2026-10-02T10:00:00Z",
        window_note="Covers 19:00 to 21:00 (this evening)",
        conditions_line="Conditions (19:00-21:00): max wind gusts 15.0 km/h",
    )

    note = "Covers 19:00 to 21:00 (this evening)"
    assert text.count(note) == 1
    assert "Conditions (19:00-21:00):" in text


def test_window_note_appears_exactly_once_in_llm_path():
    """Verify window_note is added by code and appears exactly once in LLM response."""
    note = "Covers 14:00 to midnight (rest of today)"
    llm = FakeLLMClient(
        parse_responses=[
            json.dumps({"intent": "advice", "location": "Bhopal", "activity_tags": ["cycling"], "time_ref": "today"})
        ],
        compose_responses=["High wind warning for cyclists in Bhopal: gusts reach 45.0 km/h."],
    )

    # FakeWeatherClient with window_note returned in computed facts
    class PartiallyElapsedWeatherClient(FakeWeatherClient):
        def compute_facts(self, raw_data, window_name, now_local=None):
            res = super().compute_facts(raw_data, window_name, now_local)
            res.window_note = note
            res.window_start = "14:00"
            res.window_end = "midnight"
            return res

    weather_client = PartiallyElapsedWeatherClient(facts_override={"wind_gusts": 45.0})
    graph = build_graph(weather_client=weather_client, llm=llm, checkpointer=MemorySaver())

    result = graph.invoke(
        {"query": "Is it safe to cycle in Bhopal today?"},
        config={"configurable": {"thread_id": "test-window-note-once"}},
    )

    # Compose payload sent to LLM must NOT contain the window_note
    assert len(llm.recorded_compose_payloads) == 1
    compose_payload = json.loads(llm.recorded_compose_payloads[0])
    assert "window_note" not in compose_payload

    # The final reply must have window_note added by code exactly once
    assert result["reply"].count(note) == 1
    assert "Conditions (14:00-midnight):" in result["reply"]


def test_compose_prompt_instructs_not_to_mention_window():
    """Verify compose system prompt explicitly tells LLM not to mention the window."""
    fake_llm = FakeLLMClient()
    compose_answer(fake_llm, {"activity_tags": ["cycling"]})

    # The system prompt was passed to fake_llm
    # Check that FakeLLMClient recorded or captured system prompt if we inspect
    # Let's inspect compose_answer code directly:
    import inspect
    from src import llm_tasks
    src_code = inspect.getsource(llm_tasks.compose_answer)
    assert "Do NOT mention the time window" in src_code
    assert "Do NOT mention window notes" in src_code


def test_window_end_displays_as_midnight_for_day_window():
    """Verify that end of day window displays as 'midnight' in 24-hour format."""
    client = OpenMeteoClient()
    # Mock payload with hourly data
    dt_base = datetime.datetime(2026, 10, 2, 14, 0, tzinfo=datetime.timezone.utc)
    hourly_times = [
        (dt_base + datetime.timedelta(hours=i)).strftime("%Y-%m-%dT%H:00")
        for i in range(24)
    ]
    raw = RawWeatherData(
        payload={
            "utc_offset_seconds": 0,
            "timezone": "UTC",
            "hourly": {
                "time": hourly_times,
                "wind_gusts_10m": [10.0] * 24,
                "wind_speed_10m": [5.0] * 24,
                "precipitation": [0.0] * 24,
                "precipitation_probability": [0] * 24,
                "uv_index": [1.0] * 24,
                "apparent_temperature": [20.0] * 24,
                "temperature_2m": [20.0] * 24,
                "relative_humidity_2m": [50] * 24,
                "surface_pressure": [1013.0] * 24,
                "weather_code": [0] * 24,
            },
        },
        fetch_time="2026-10-02T14:00:00Z",
    )

    now_14 = datetime.datetime(2026, 10, 2, 14, 15, tzinfo=datetime.timezone.utc)
    computed = client.compute_facts(raw, "today", now_local=now_14)

    assert computed.window_start == "14:00"
    assert computed.window_end == "midnight"
    assert computed.window_note == "Covers 14:00 to midnight (rest of today)"


def test_conflict_resolver_reasons_single_and_clear():
    """Verify resolver reasons for single matching SOP and clear SOP."""
    from src.conflict_resolver import resolve

    # 1. Single matching hazard SOP
    sop_hazard = SOPResult(
        sop_id="EXR-001",
        match_type="numeric",
        effective_severity="high",
        priority=10,
        rendered_advice="High wind",
        facts_used={},
        status="matched",
    )
    res_single = resolve([sop_hazard])
    assert res_single.reason == "Only EXR-001 applied"

    # 2. Clear SOP
    sop_clear = SOPResult(
        sop_id="CLR-EXR",
        match_type="clear",
        effective_severity="info",
        priority=100,
        rendered_advice="Conditions clear",
        facts_used={},
        status="matched",
    )
    res_clear = resolve([sop_clear])
    assert res_clear.reason == "No hazard SOP matched; clear baseline applies"


def test_window_note_reworded_formats():
    """Verify window note descriptions for all window types."""
    client = OpenMeteoClient()
    dt_base = datetime.datetime(2026, 10, 2, 0, 0, tzinfo=datetime.timezone.utc)
    hourly_times = [
        (dt_base + datetime.timedelta(hours=i)).strftime("%Y-%m-%dT%H:00")
        for i in range(72)
    ]
    raw = RawWeatherData(
        payload={
            "utc_offset_seconds": 0,
            "timezone": "UTC",
            "hourly": {
                "time": hourly_times,
                "wind_gusts_10m": [10.0] * 72,
                "wind_speed_10m": [5.0] * 72,
                "precipitation": [0.0] * 72,
                "precipitation_probability": [0] * 72,
                "uv_index": [1.0] * 72,
                "apparent_temperature": [20.0] * 72,
                "temperature_2m": [20.0] * 72,
                "relative_humidity_2m": [50] * 72,
                "surface_pressure": [1013.0] * 72,
                "weather_code": [0] * 72,
            },
        },
        fetch_time="2026-10-02T10:00:00Z",
    )

    now_local = datetime.datetime(2026, 10, 2, 10, 0, tzinfo=datetime.timezone.utc)

    c_today = client.compute_facts(raw, "today", now_local=now_local)
    assert c_today.window_note == "Covers 10:00 to midnight (rest of today)"

    c_evening = client.compute_facts(raw, "this_evening", now_local=now_local)
    assert c_evening.window_note == "Covers 17:00 to 21:00 (this evening)"

    c_tomorrow = client.compute_facts(raw, "tomorrow", now_local=now_local)
    assert c_tomorrow.window_note == "Covers 06:00 to 22:00 (tomorrow daytime)"

    c_now = client.compute_facts(raw, "now", now_local=now_local)
    assert c_now.window_note == "Covers 10:00 to 14:00 (next 3 hours)"
