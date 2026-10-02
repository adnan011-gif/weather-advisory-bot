"""Tests for dynamic SOP freshness and error handling:
1. Load SOPs from disk on every turn.
2. Temp SOP added between turns fires immediately without restart.
3. Malformed SOP file gives clear user-visible error naming the file.
4. answer_source, reason, and model_used are recorded in state and decision_log.
"""

from __future__ import annotations

import json
from pathlib import Path
from langgraph.checkpoint.memory import MemorySaver

from src.graph import build_graph
from tests.test_graph_flow import FakeLLMClient, FakeWeatherClient


def test_dynamic_sop_loading_between_turns(tmp_path: Path):
    """Verify that adding an SOP file between turns immediately takes effect on the next turn."""
    # Create temp sops dir with standard real SOPs copied or minimal SOP
    sops_dir = tmp_path / "sops"
    sops_dir.mkdir()

    # Base SOP: CLR-EXR for cycling
    (sops_dir / "CLR-EXR.yaml").write_text(
        """id: CLR-EXR
title: Exercise Clear
category: clear
severity: info
priority: 1
applies_to:
  - cycling
match_type: clear
advice: Conditions are clear for cycling.
rationale: Safe.
""",
        encoding="utf-8",
    )

    llm = FakeLLMClient(
        parse_responses=[
            # Turn 1: user asks about cycling
            json.dumps({"intent": "advice", "location": "Bhopal", "activity_tags": ["cycling"], "time_ref": "today"}),
            # Turn 2: user asks about gardening
            json.dumps({"intent": "advice", "location": "Bhopal", "activity_tags": ["gardening"], "time_ref": "today"}),
        ],
        compose_responses=[
            "High winds detected (25.0 km/h). Secure lightweight plants in Bhopal.",
        ],
    )

    weather_client = FakeWeatherClient(facts_override={"wind_gusts": 25.0})
    graph = build_graph(
        weather_client=weather_client,
        llm=llm,
        sop_dir=sops_dir,
        checkpointer=MemorySaver(),
    )

    # Turn 1: cycling
    r1 = graph.invoke(
        {"query": "Can I cycle in Bhopal today?"},
        config={"configurable": {"thread_id": "freshness-thread"}},
    )
    assert "CLR-EXR" in r1["reply"]

    # Now add a new SOP dynamically on disk
    (sops_dir / "GAR-001.yaml").write_text(
        """id: GAR-001
title: Gardening Wind Advisory
category: wind
severity: moderate
priority: 70
applies_to:
  - gardening
match_type: numeric
condition:
  fact: wind_gusts
  op: ">"
  value: 20
advice: High winds detected ({wind_gusts} km/h). Secure lightweight plants.
rationale: Wind risk for gardening.
""",
        encoding="utf-8",
    )

    # Turn 2: gardening (same graph and session)
    r2 = graph.invoke(
        {"query": "Can I do gardening in Bhopal today?"},
        config={"configurable": {"thread_id": "freshness-thread"}},
    )

    # Must match GAR-001 without restarting or rebuilding the graph
    assert "GAR-001" in r2["reply"]
    assert "Secure lightweight plants" in r2["reply"]


def test_malformed_sop_gives_clear_user_visible_error_naming_file(tmp_path: Path):
    """Verify that a malformed SOP YAML yields a user-visible error naming the offending file."""
    sops_dir = tmp_path / "sops"
    sops_dir.mkdir()

    # Broken SOP
    broken_file = sops_dir / "BROKEN-001.yaml"
    broken_file.write_text("id: BROKEN-001\nseverity: unknown_sev_bad_enum\n", encoding="utf-8")

    fake_llm = FakeLLMClient()
    graph = build_graph(
        weather_client=FakeWeatherClient(),
        llm=fake_llm,
        sop_dir=sops_dir,
        checkpointer=MemorySaver(),
    )

    res = graph.invoke(
        {"query": "Is it safe to run in Bhopal?"},
        config={"configurable": {"thread_id": "test-malformed-sop"}},
    )

    assert res["kind"] == "format_error"
    assert "BROKEN-001.yaml" in res["reply"]
    assert "Policy configuration error:" in res["reply"]


def test_answer_source_reason_and_model_used_in_state_and_log():
    """Verify answer_source, reason, and model_used are recorded in state and decision_log."""
    llm = FakeLLMClient(
        parse_responses=[
            json.dumps({"intent": "advice", "location": "Bhopal", "activity_tags": ["cycling"], "time_ref": "today"})
        ],
        model_used="gemini-2.5-flash-test",
    )
    graph = build_graph(weather_client=FakeWeatherClient(), llm=llm, checkpointer=MemorySaver())
    res = graph.invoke(
        {"query": "Is it safe to cycle in Bhopal today?"},
        config={"configurable": {"thread_id": "test-audit-fields"}},
    )

    # State fields
    assert res["answer_source"] == "template"
    assert res["reason"] == "no_llm_needed"
    assert res["model_used"] == "gemini-2.5-flash-test"

    # Decision log entry
    assert len(res["decision_log"]) >= 1
    last_entry = res["decision_log"][-1]
    assert last_entry["answer_source"] == "template"
    assert last_entry["reason"] == "no_llm_needed"
    assert last_entry["model_used"] == "gemini-2.5-flash-test"
