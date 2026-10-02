"""End-to-end and branch coverage tests for the LangGraph workflow.

Tests session continuity, prompt isolation, allow-list enforcement,
deterministic fallbacks, and all conditional graph routing branches using fake clients.
"""

from __future__ import annotations

import json
from langgraph.checkpoint.memory import MemorySaver

from src.graph import build_graph
from src.llm_client import LLMError
from src.weather_client import (
    GeocodeResult,
    RawWeatherData,
    ComputedFactsResult,
    LocationError,
    WeatherAPIError,
    WindowPassedError,
)


class FakeLLMClient:
    """Scriptable mock LLM client for tests."""

    def __init__(
        self,
        parse_responses: list[str | Exception] | None = None,
        compose_responses: list[str | Exception] | None = None,
        model_used: str | None = "fake-primary-model",
    ) -> None:
        self.parse_responses = list(parse_responses or [])
        self.compose_responses = list(compose_responses or [])
        self.recorded_compose_payloads: list[str] = []
        self.recorded_parse_calls: list[str] = []
        self.model_used = model_used

    def parse_intent_raw(self, system: str, user_text: str) -> str:
        self.recorded_parse_calls.append(user_text)
        if self.parse_responses:
            item = self.parse_responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        # Default response
        return json.dumps({
            "intent": "advice",
            "location": "Bhopal",
            "activity_tags": ["cycling"],
            "time_ref": "today",
            "cited_sop_ids": [],
        })

    def compose_raw(self, system: str, payload_json: str) -> str:
        self.recorded_compose_payloads.append(payload_json)
        if self.compose_responses:
            item = self.compose_responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return "Advisory for Bhopal, Madhya Pradesh, India: Cycling conditions are clear and safe."


class FakeWeatherClient:
    """Scriptable mock weather client for tests."""

    def __init__(
        self,
        geocode_error: Exception | None = None,
        weather_error: Exception | None = None,
        window_error: Exception | None = None,
        facts_override: dict | None = None,
    ) -> None:
        self.geocode_error = geocode_error
        self.weather_error = weather_error
        self.window_error = window_error
        self.facts_override = facts_override

    def geocode(self, city: str) -> GeocodeResult:
        if self.geocode_error:
            raise self.geocode_error
        return GeocodeResult(
            lat=23.25,
            lon=77.40,
            resolved_name=f"{city}, India",
            timezone="Asia/Kolkata",
        )

    def fetch_weather(self, lat: float, lon: float) -> RawWeatherData:
        if self.weather_error:
            raise self.weather_error
        return RawWeatherData(
            payload={"utc_offset_seconds": 19800, "timezone": "Asia/Kolkata"},
            fetch_time="2026-10-02T10:00:00Z",
        )

    def compute_facts(
        self,
        raw_data: RawWeatherData | dict,
        window_name: str,
        now_local=None,
    ) -> ComputedFactsResult:
        if self.window_error:
            raise self.window_error

        # Default benign facts
        default_facts = {
            "temperature": 24.0,
            "apparent_temperature": 24.0,
            "wind_speed": 10.0,
            "wind_gusts": 15.0,
            "precipitation_sum": 0.0,
            "precipitation_probability_max": 10.0,
            "uv_index_max": 4.0,
            "weather_code": [0],
            "pressure_msl": 1015.0,
            "humidity": 50.0,
        }
        if self.facts_override:
            default_facts.update(self.facts_override)

        return ComputedFactsResult(
            facts=default_facts,
            window_name=window_name,
            partly_passed=False,
            fetch_time="2026-10-02T10:00:00Z",
            timezone="Asia/Kolkata",
            utc_offset_seconds=19800,
        )


def test_out_of_vocabulary_tag_dropped():
    """Verify that an activity tag not present in derived vocabulary is strictly dropped."""
    fake_llm = FakeLLMClient(
        parse_responses=[
            json.dumps({
                "intent": "advice",
                "location": "Bhopal",
                "activity_tags": ["cycling", "deep_sea_scuba_diving_unsupported"],
                "time_ref": "today",
            })
        ]
    )
    graph = build_graph(weather_client=FakeWeatherClient(), llm=fake_llm, checkpointer=MemorySaver())
    res = graph.invoke({"query": "Can I go scuba diving and cycling in Bhopal?"}, config={"configurable": {"thread_id": "t1"}})

    # Only 'cycling' should survive in activity_tags
    assert res["activity_tags"] == ["cycling"]
    assert "deep_sea_scuba_diving_unsupported" not in res["activity_tags"]


def test_llm_exception_triggers_templated_answer():
    """Verify that when LLM composer throws an exception, templated answer fallback fires cleanly."""
    fake_llm = FakeLLMClient(
        compose_responses=[LLMError("Simulated LLM API rate limit 429")]
    )
    graph = build_graph(weather_client=FakeWeatherClient(), llm=fake_llm, checkpointer=MemorySaver())
    res = graph.invoke({"query": "Cycling in Bhopal today?"}, config={"configurable": {"thread_id": "t2"}})

    assert res["kind"] == "advice"
    assert "Policy: CLR-EXR (info)" in res["reply"]
    assert "No policy warning threshold was triggered" in res["reply"]


def test_routing_branches():
    """Test every conditional routing branch:
    - parse_failed
    - ask_location
    - ask_activity
    - format_error (geocode error)
    - format_error (weather error)
    - window_passed
    - data_incomplete
    - no_guidance
    - explain with empty log
    - explain with a fake id
    """
    # 1. parse_failed: LLM returns completely broken JSON
    bad_llm = FakeLLMClient(parse_responses=["NOT_JSON_AT_ALL", "STILL_NOT_JSON"])
    g1 = build_graph(weather_client=FakeWeatherClient(), llm=bad_llm, checkpointer=MemorySaver())
    r1 = g1.invoke({"query": "???"}, config={"configurable": {"thread_id": "b1"}})
    assert r1["kind"] == "parse_failed"
    assert "couldn't understand" in r1["reply"]

    # 2. ask_location: has activity 'hiking', but location is None
    loc_missing_llm = FakeLLMClient(parse_responses=[
        json.dumps({"intent": "advice", "location": None, "activity_tags": ["hiking"], "time_ref": "today"})
    ])
    g2 = build_graph(weather_client=FakeWeatherClient(), llm=loc_missing_llm, checkpointer=MemorySaver())
    r2 = g2.invoke({"query": "Can I go hiking?"}, config={"configurable": {"thread_id": "b2"}})
    assert r2["kind"] == "ask_location"
    assert "Please specify the city" in r2["reply"]

    # 3. ask_activity: has location 'Paris', but activity is empty
    act_missing_llm = FakeLLMClient(parse_responses=[
        json.dumps({"intent": "advice", "location": "Paris", "activity_tags": [], "time_ref": "today"})
    ])
    g3 = build_graph(weather_client=FakeWeatherClient(), llm=act_missing_llm, checkpointer=MemorySaver())
    r3 = g3.invoke({"query": "What is the weather in Paris?"}, config={"configurable": {"thread_id": "b3"}})
    assert r3["kind"] == "ask_activity"
    assert "Please specify the outdoor activity" in r3["reply"]

    # 4. format_error via geocode failure
    geo_err_weather = FakeWeatherClient(geocode_error=LocationError("City not found"))
    g4 = build_graph(weather_client=geo_err_weather, llm=FakeLLMClient(), checkpointer=MemorySaver())
    r4 = g4.invoke({"query": "Cycling in NonExistentCity today?"}, config={"configurable": {"thread_id": "b4"}})
    assert r4["kind"] == "format_error"
    assert "Could not find coordinates" in r4["reply"]

    # 5. format_error via weather API failure
    api_err_weather = FakeWeatherClient(weather_error=WeatherAPIError("Open-Meteo HTTP 503"))
    g5 = build_graph(weather_client=api_err_weather, llm=FakeLLMClient(), checkpointer=MemorySaver())
    r5 = g5.invoke({"query": "Cycling in Bhopal today?"}, config={"configurable": {"thread_id": "b5"}})
    assert r5["kind"] == "format_error"
    assert "Weather forecast service is temporarily unavailable" in r5["reply"]

    # 6. window_passed
    win_err_weather = FakeWeatherClient(window_error=WindowPassedError("Window 'this_evening' has already ended."))
    g6 = build_graph(weather_client=win_err_weather, llm=FakeLLMClient(), checkpointer=MemorySaver())
    r6 = g6.invoke({"query": "Cycling in Bhopal this evening?"}, config={"configurable": {"thread_id": "b6"}})
    assert r6["kind"] == "window_passed"
    assert "has already ended" in r6["reply"]

    # 7. data_incomplete: 0 matched, but missing facts
    data_inc_weather = FakeWeatherClient(facts_override={"temperature": None, "apparent_temperature": None, "uv_index_max": None, "wind_gusts": None})
    g7 = build_graph(weather_client=data_inc_weather, llm=FakeLLMClient(), checkpointer=MemorySaver())
    r7 = g7.invoke({"query": "Running in Bhopal today?"}, config={"configurable": {"thread_id": "b7"}})
    assert r7["kind"] == "data_incomplete"
    assert "incomplete" in r7["reply"]

    # 8. no_guidance: intent is out_of_scope
    oos_llm = FakeLLMClient(parse_responses=[
        json.dumps({"intent": "out_of_scope", "location": None, "activity_tags": [], "time_ref": None})
    ])
    g8 = build_graph(weather_client=FakeWeatherClient(), llm=oos_llm, checkpointer=MemorySaver())
    r8 = g8.invoke({"query": "Write a python script to sort a list"}, config={"configurable": {"thread_id": "b8"}})
    assert r8["kind"] == "no_guidance"

    # 9. explain with empty log
    exp_empty_llm = FakeLLMClient(parse_responses=[
        json.dumps({"intent": "explain", "location": None, "activity_tags": [], "time_ref": None, "cited_sop_ids": []})
    ])
    g9 = build_graph(weather_client=FakeWeatherClient(), llm=exp_empty_llm, checkpointer=MemorySaver())
    r9 = g9.invoke({"query": "Why did you say that?"}, config={"configurable": {"thread_id": "b9"}})
    assert r9["kind"] == "explain"
    assert "nothing to explain yet" in r9["reply"]

    # 10. explain with a fake SOP ID
    exp_fake_llm = FakeLLMClient(parse_responses=[
        json.dumps({"intent": "explain", "location": None, "activity_tags": [], "time_ref": None, "cited_sop_ids": ["FAKE-999"]})
    ])
    # Put a dummy entry in log first
    mem = MemorySaver()
    g10 = build_graph(weather_client=FakeWeatherClient(), llm=FakeLLMClient(), checkpointer=mem)
    # Turn 1: generate an advisory so log exists
    g10.invoke({"query": "Cycling in Bhopal today?"}, config={"configurable": {"thread_id": "b10"}})
    # Turn 2: ask about fake id
    g10_fake = build_graph(weather_client=FakeWeatherClient(), llm=exp_fake_llm, checkpointer=mem)
    r10 = g10_fake.invoke({"query": "Why did you use FAKE-999?"}, config={"configurable": {"thread_id": "b10"}})
    assert r10["kind"] == "explain"
    assert "SOP 'FAKE-999' does not exist in our policy registry" in r10["reply"]


def test_three_turn_session_flow():
    """Verify a complete 3-turn interactive conversation with session memory:
    Turn 1: "cycling in Bhopal today?"
    Turn 2: "what about this evening?" (reuses Bhopal and cycling)
    Turn 3: "why did you say that?" (explains from recorded decision log)
    """
    memory = MemorySaver()
    thread_config = {"configurable": {"thread_id": "multi-turn-session"}}

    # Turn 1: Advice for cycling in Bhopal today
    llm_turn1 = FakeLLMClient(
        parse_responses=[
            json.dumps({"intent": "advice", "location": "Bhopal", "activity_tags": ["cycling"], "time_ref": "today", "cited_sop_ids": []})
        ],
        compose_responses=[
            "Under CLR-EXR, cycling conditions in Bhopal today are clear and meet safety thresholds."
        ],
    )
    g1 = build_graph(weather_client=FakeWeatherClient(), llm=llm_turn1, checkpointer=memory)
    r1 = g1.invoke({"query": "cycling in Bhopal today?"}, config=thread_config)
    assert r1["location"] == "Bhopal"
    assert r1["activity_tags"] == ["cycling"]
    assert r1["time_ref"] == "today"
    assert "Policy: CLR-EXR" in r1["reply"]

    # Turn 2: "what about this evening?" -> Omission of location & activity
    llm_turn2 = FakeLLMClient(
        parse_responses=[
            json.dumps({"intent": "advice", "location": None, "activity_tags": [], "time_ref": "this_evening", "cited_sop_ids": []})
        ],
        compose_responses=[
            "For this evening in Bhopal, wind gusts reach 42.0 km/h under EXR-003."
        ],
    )
    # Simulate high wind gusts this evening to trigger EXR-003
    weather_turn2 = FakeWeatherClient(facts_override={"wind_gusts": 42.0})
    g2 = build_graph(weather_client=weather_turn2, llm=llm_turn2, checkpointer=memory)
    r2 = g2.invoke({"query": "what about this evening?"}, config=thread_config)

    # Reused saved context
    assert r2["location"] == "Bhopal"
    assert r2["activity_tags"] == ["cycling"]
    assert r2["time_ref"] == "this_evening"
    assert "Policy: EXR-003" in r2["reply"]

    # Turn 3: "why did you say that?" -> Explain decision
    llm_turn3 = FakeLLMClient(
        parse_responses=[
            json.dumps({"intent": "explain", "location": None, "activity_tags": [], "time_ref": None, "cited_sop_ids": []})
        ]
    )
    g3 = build_graph(weather_client=FakeWeatherClient(), llm=llm_turn3, checkpointer=memory)
    r3 = g3.invoke({"query": "why did you say that?"}, config=thread_config)

    assert r3["kind"] == "explain"
    assert "EXR-003" in r3["reply"]
    assert "wind_gusts=42.0" in r3["reply"]


def test_raw_user_message_never_appears_in_compose_payload():
    """Verify that the raw query text is never passed into the compose payload."""
    secret_marker = "DO_NOT_LEAK_USER_SECRET_QUERY_TEXT"
    raw_query = f"Can I go cycling in Bhopal? {secret_marker}"

    llm = FakeLLMClient(
        parse_responses=[
            json.dumps({"intent": "advice", "location": "Bhopal", "activity_tags": ["cycling"], "time_ref": "today"})
        ]
    )
    weather_client = FakeWeatherClient(facts_override={"wind_gusts": 42.0})
    graph = build_graph(weather_client=weather_client, llm=llm, checkpointer=MemorySaver())
    graph.invoke({"query": raw_query}, config={"configurable": {"thread_id": "payload-isolation-test"}})

    # Check recorded compose payload
    assert len(llm.recorded_compose_payloads) == 1
    compose_payload = llm.recorded_compose_payloads[0]
    assert secret_marker not in compose_payload
    assert raw_query not in compose_payload
