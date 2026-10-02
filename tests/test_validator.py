"""Unit tests for validator, citation footer builder, and templated answer generator."""

from __future__ import annotations

from src.validator import validate_reply, build_footer, templated_answer
from src.sop_engine import SOPResult


def test_invented_number_caught():
    """Verify that any number not present in the allow-list is rejected."""
    allowed_numbers = {20.0, 30.0, 1.0}
    known_ids = {"EXR-001", "EXR-002"}

    # Allowed number: 20.0
    valid_reply = "Temperature reaches 20.0°C. Be careful."
    ok, err = validate_reply(valid_reply, allowed_numbers, known_ids)
    assert ok is True
    assert err is None

    # Hallucinated number: 25.5 is not in allowed_numbers
    hallucinated_reply = "Temperature reaches 25.5°C. Please stay safe."
    ok, err = validate_reply(hallucinated_reply, allowed_numbers, known_ids)
    assert ok is False
    assert "Unallowed number in reply: 25.5" in err


def test_fake_sop_id_caught():
    """Verify that citing an unrecognized SOP ID fails validation."""
    allowed_numbers = {10.0, 20.0}
    known_ids = {"EXR-001", "EXR-002", "CLR-EXR"}

    # Known ID
    valid_reply = "According to EXR-001, take shelter."
    ok, err = validate_reply(valid_reply, allowed_numbers, known_ids)
    assert ok is True

    # Fake ID
    fake_reply = "According to EXR-999, conditions are hazardous."
    ok, err = validate_reply(fake_reply, allowed_numbers, known_ids)
    assert ok is False
    assert "Cited unknown SOP ID: 'EXR-999'" in err


def test_sop_id_numbers_not_flagged_as_unallowed_numbers():
    """Verify that digits inside an SOP ID (e.g. 001 in EXR-001) are not treated as rogue numbers."""
    allowed_numbers = {30.0}
    known_ids = {"EXR-001"}

    reply = "Under EXR-001, temperature reaches 30.0°C."
    ok, err = validate_reply(reply, allowed_numbers, known_ids)
    assert ok is True


def test_build_footer_and_templated_answer():
    """Verify citation footer formatting and deterministic templated answer creation."""
    primary = SOPResult(
        sop_id="EXR-002",
        status="matched",
        effective_severity="high",
        rendered_advice="Perceived temperature reaches 41.0°C. Avoid outdoor exertion.",
        facts_used={"apparent_temperature": 41.0},
        override=False,
        priority=5,
        category="outdoor_exercise",
        match_type="numeric",
    )
    also_applies = [
        SOPResult(
            sop_id="EXR-001",
            status="matched",
            effective_severity="moderate",
            rendered_advice="Peak UV reaches 8.5. Wear sunscreen.",
            facts_used={"uv_index_max": 8.5},
            override=False,
            priority=10,
            category="outdoor_exercise",
            match_type="numeric",
        )
    ]

    footer = build_footer(
        primary=primary,
        also_applies=also_applies,
        resolved_name="Bhopal, Madhya Pradesh, India",
        fetch_time="2026-10-02T10:00:00Z",
        skipped_ids=["EXR-003"],
    )
    assert "Policy: EXR-002 (high)" in footer
    assert "Also applies: EXR-001 (moderate)" in footer
    assert "1 safety checks could not be run (missing data): EXR-003" in footer
    assert "Location: Bhopal, Madhya Pradesh, India" in footer

    # Templated answer includes advice and footer
    answer = templated_answer(
        primary=primary,
        also_applies=also_applies,
        resolved_name="Bhopal, Madhya Pradesh, India",
        time_ref="today",
        fetch_time="2026-10-02T10:00:00Z",
        skipped_ids=["EXR-003"],
        window_note="Window 'today' evaluated for remainder of day",
        assumed_time=True,
    )
    assert "Advisory for Bhopal, Madhya Pradesh, India (assuming time window 'today'):" in answer
    assert "Perceived temperature reaches 41.0°C" in answer
    assert "Policy: EXR-002 (high)" in answer
