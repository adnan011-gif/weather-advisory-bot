"""Unit and integration tests for real production SOPs in sops/.

Validates schema loading against config/facts.yaml, boundary conditions,
semantic rubric tier progression, override dominance, clear SOP gating,
placeholder substitution completeness, and dynamic vocabulary extraction.
"""

from __future__ import annotations

import re
from pathlib import Path
import pytest

from src.facts_registry import FactsRegistry
from src.sop_loader import load_sops, derive_vocabulary
from src.sop_engine import evaluate_all, evaluate_sop
from src.conflict_resolver import resolve


@pytest.fixture(scope="module")
def facts_registry() -> FactsRegistry:
    return FactsRegistry()


@pytest.fixture(scope="module")
def real_sops(facts_registry: FactsRegistry) -> list:
    sops_dir = Path(__file__).resolve().parent.parent / "sops"
    return load_sops(sops_dir, facts_registry)


@pytest.fixture(scope="module")
def sops_by_id(real_sops: list) -> dict:
    return {s.id: s for s in real_sops}


@pytest.fixture
def benign_facts() -> dict:
    """Fixture with benign, clear meteorological conditions."""
    return {
        "temperature": 22.0,
        "apparent_temperature": 22.0,
        "wind_speed": 10.0,
        "wind_gusts": 15.0,
        "precipitation_sum": 0.0,
        "precipitation_probability_max": 10.0,
        "uv_index_max": 4.0,
        "weather_code": [0],
        "pressure_msl": 1015.0,
        "humidity": 50.0,
    }


def test_load_all_real_sops(real_sops: list):
    """Verify all 16 SOPs (12 hazard/advice + 4 clear baseline) load without error."""
    assert len(real_sops) == 16
    ids = {s.id for s in real_sops}
    expected_ids = {
        "EXR-001", "EXR-002", "EXR-003",
        "TRV-001", "TRV-002", "TRV-003",
        "VUL-001", "VUL-002", "VUL-003",
        "LSR-001", "GEN-001", "GEN-002",
        "CLR-EXR", "CLR-TRV", "CLR-VUL", "CLR-LSR",
    }
    assert ids == expected_ids


def test_vocabulary_tags_equal_exact_list(real_sops: list):
    """Verify vocabulary derived from applies_to tags matches the exact project tag set."""
    vocab = derive_vocabulary(real_sops)
    expected_tags = {
        "running", "cycling", "hiking", "outdoor_exercise",
        "two_wheeler", "commute", "driving", "travel",
        "children", "elderly", "pets", "picnic", "outdoor_event",
    }
    assert set(vocab.keys()) == expected_tags


def test_boundary_conditions_for_hazard_sops(sops_by_id: dict, benign_facts: dict):
    """Test boundary triggers (> vs >=) for all individual hazard SOPs."""
    # EXR-001: uv_index_max >= 8.0
    sop = sops_by_id["EXR-001"]
    facts_below = {**benign_facts, "uv_index_max": 7.9}
    facts_at = {**benign_facts, "uv_index_max": 8.0}
    assert evaluate_sop(sop, facts_below, ["running"]).status == "not_matched"
    assert evaluate_sop(sop, facts_at, ["running"]).status == "matched"

    # EXR-002: apparent_temperature >= 40.0
    sop = sops_by_id["EXR-002"]
    facts_below = {**benign_facts, "apparent_temperature": 39.9}
    facts_at = {**benign_facts, "apparent_temperature": 40.0}
    assert evaluate_sop(sop, facts_below, ["running"]).status == "not_matched"
    assert evaluate_sop(sop, facts_at, ["running"]).status == "matched"

    # EXR-003: wind_gusts >= 40.0 (cycling, two_wheeler)
    sop = sops_by_id["EXR-003"]
    facts_below = {**benign_facts, "wind_gusts": 39.9}
    facts_at = {**benign_facts, "wind_gusts": 40.0}
    assert evaluate_sop(sop, facts_below, ["cycling"]).status == "not_matched"
    assert evaluate_sop(sop, facts_at, ["cycling"]).status == "matched"

    # TRV-001: precipitation_probability_max >= 70.0
    sop = sops_by_id["TRV-001"]
    facts_below = {**benign_facts, "precipitation_probability_max": 69.9}
    facts_at = {**benign_facts, "precipitation_probability_max": 70.0}
    assert evaluate_sop(sop, facts_below, ["driving"]).status == "not_matched"
    assert evaluate_sop(sop, facts_at, ["driving"]).status == "matched"

    # TRV-002: precipitation_sum >= 5.0
    sop = sops_by_id["TRV-002"]
    facts_below = {**benign_facts, "precipitation_sum": 4.9}
    facts_at = {**benign_facts, "precipitation_sum": 5.0}
    assert evaluate_sop(sop, facts_below, ["two_wheeler"]).status == "not_matched"
    assert evaluate_sop(sop, facts_at, ["two_wheeler"]).status == "matched"

    # TRV-003: wind_gusts >= 60.0
    sop = sops_by_id["TRV-003"]
    facts_below = {**benign_facts, "wind_gusts": 59.9}
    facts_at = {**benign_facts, "wind_gusts": 60.0}
    assert evaluate_sop(sop, facts_below, ["travel"]).status == "not_matched"
    assert evaluate_sop(sop, facts_at, ["travel"]).status == "matched"

    # VUL-001: apparent_temperature >= 35.0 OR uv_index_max >= 6.0
    sop = sops_by_id["VUL-001"]
    facts_below = {**benign_facts, "apparent_temperature": 34.9, "uv_index_max": 5.9}
    facts_temp = {**benign_facts, "apparent_temperature": 35.0, "uv_index_max": 3.0}
    facts_uv = {**benign_facts, "apparent_temperature": 25.0, "uv_index_max": 6.0}
    assert evaluate_sop(sop, facts_below, ["children"]).status == "not_matched"
    assert evaluate_sop(sop, facts_temp, ["children"]).status == "matched"
    assert evaluate_sop(sop, facts_uv, ["children"]).status == "matched"

    # VUL-002: apparent_temperature >= 36.0
    sop = sops_by_id["VUL-002"]
    facts_below = {**benign_facts, "apparent_temperature": 35.9}
    facts_at = {**benign_facts, "apparent_temperature": 36.0}
    assert evaluate_sop(sop, facts_below, ["elderly"]).status == "not_matched"
    assert evaluate_sop(sop, facts_at, ["elderly"]).status == "matched"

    # VUL-003: temperature >= 33.0
    sop = sops_by_id["VUL-003"]
    facts_below = {**benign_facts, "temperature": 32.9}
    facts_at = {**benign_facts, "temperature": 33.0}
    assert evaluate_sop(sop, facts_below, ["pets"]).status == "not_matched"
    assert evaluate_sop(sop, facts_at, ["pets"]).status == "matched"

    # GEN-001: weather_code intersects [95, 96, 99]
    sop = sops_by_id["GEN-001"]
    facts_no_thunder = {**benign_facts, "weather_code": [0, 1, 2, 61]}
    facts_thunder = {**benign_facts, "weather_code": [0, 95]}
    assert evaluate_sop(sop, facts_no_thunder, ["any"]).status == "not_matched"
    assert evaluate_sop(sop, facts_thunder, ["any"]).status == "matched"

    # GEN-002: (precip_sum >= 64.5 OR code in [65, 82, 95, 96, 99]) AND pressure <= 1004.0
    sop = sops_by_id["GEN-002"]
    facts_heavy_rain = {**benign_facts, "precipitation_sum": 64.5, "pressure_msl": 1004.0}
    facts_high_pressure = {**benign_facts, "precipitation_sum": 70.0, "pressure_msl": 1004.1}
    facts_low_rain = {**benign_facts, "precipitation_sum": 64.4, "pressure_msl": 1000.0, "weather_code": [0]}
    assert evaluate_sop(sop, facts_heavy_rain, ["any"]).status == "matched"
    assert evaluate_sop(sop, facts_high_pressure, ["any"]).status == "not_matched"
    assert evaluate_sop(sop, facts_low_rain, ["any"]).status == "not_matched"


def test_picnic_tiers_poor_fair_good(sops_by_id: dict, benign_facts: dict):
    """Test LSR-001 semantic rubric tier progression: poor, fair, and good."""
    sop = sops_by_id["LSR-001"]

    # Tier: poor (precip prob >= 50 or gusts >= 40, etc.)
    facts_poor = {**benign_facts, "precipitation_probability_max": 55.0}
    res_poor = evaluate_sop(sop, facts_poor, ["picnic"])
    assert res_poor.status == "matched"
    assert res_poor.tier == "poor"
    assert res_poor.effective_severity == "moderate"

    # Tier: fair (precip prob >= 30, gusts >= 30, etc.)
    facts_fair = {
        **benign_facts,
        "precipitation_probability_max": 35.0,
        "precipitation_sum": 0.5,
        "wind_gusts": 20.0,
        "apparent_temperature": 26.0,
        "uv_index_max": 5.0,
    }
    res_fair = evaluate_sop(sop, facts_fair, ["picnic"])
    assert res_fair.status == "matched"
    assert res_fair.tier == "fair"
    assert res_fair.effective_severity == "low"

    # Tier: good (all conditions explicitly below thresholds)
    facts_good = {
        **benign_facts,
        "precipitation_probability_max": 15.0,
        "precipitation_sum": 0.0,
        "wind_gusts": 15.0,
        "apparent_temperature": 24.0,
        "uv_index_max": 4.0,
    }
    res_good = evaluate_sop(sop, facts_good, ["picnic"])
    assert res_good.status == "matched"
    assert res_good.tier == "good"
    assert res_good.effective_severity == "info"

    # Missing fact makes good tier unevaluable, not a catch-all!
    facts_missing = {
        **benign_facts,
        "precipitation_probability_max": 15.0,
        "precipitation_sum": 0.0,
        "wind_gusts": 15.0,
        "apparent_temperature": None,  # Missing
        "uv_index_max": 4.0,
    }
    res_missing = evaluate_sop(sop, facts_missing, ["picnic"])
    assert res_missing.status == "unevaluable"


def test_gen_002_wins_via_override(real_sops: list, benign_facts: dict):
    """Test that GEN-002 wins conflict resolution over all other matching critical SOPs."""
    facts = {
        **benign_facts,
        "precipitation_sum": 80.0,
        "pressure_msl": 995.0,
        "weather_code": [95],  # Triggers GEN-001
        "apparent_temperature": 42.0,  # Triggers EXR-002
        "wind_gusts": 65.0,  # Triggers TRV-003
    }
    results, matched_ids, _ = evaluate_all(real_sops, facts, ["running", "travel", "driving"])
    assert "GEN-002" in matched_ids
    assert "GEN-001" in matched_ids
    assert "EXR-002" in matched_ids

    resolution = resolve([r for r in results if r.status == "matched"])
    assert resolution.primary is not None
    assert resolution.primary.sop_id == "GEN-002"
    assert "override" in resolution.reason


def test_clear_sops_behavior(real_sops: list, benign_facts: dict):
    """Test that clear SOPs fire on benign facts, and are blocked by GEN-001 and null facts."""
    # 1. Benign facts: Clear SOPs fire for their respective activity categories
    results_exr, matched_exr, _ = evaluate_all(real_sops, benign_facts, ["running"])
    assert "CLR-EXR" in matched_exr

    results_trv, matched_trv, _ = evaluate_all(real_sops, benign_facts, ["commute"])
    assert "CLR-TRV" in matched_trv

    results_vul, matched_vul, _ = evaluate_all(real_sops, benign_facts, ["children"])
    assert "CLR-VUL" in matched_vul

    results_lsr, matched_lsr, _ = evaluate_all(real_sops, benign_facts, ["picnic"])
    # In benign conditions, LSR-001 matches tier 'good', CLR-LSR is baseline
    assert "LSR-001" in matched_lsr

    # 2. Blocked by GEN-001 (thunderstorm code 95)
    facts_thunder = {**benign_facts, "weather_code": [95]}
    results_thun, matched_thun, _ = evaluate_all(real_sops, facts_thunder, ["running", "commute", "children"])
    assert "GEN-001" in matched_thun
    assert "CLR-EXR" not in matched_thun
    assert "CLR-TRV" not in matched_thun
    assert "CLR-VUL" not in matched_thun

    # 3. Blocked by null facts in general or category hazard
    facts_null = {**benign_facts, "weather_code": None}
    results_null, matched_null, uneval_null = evaluate_all(real_sops, facts_null, ["running"])
    assert "GEN-001" in uneval_null
    assert "CLR-EXR" in uneval_null
    assert "CLR-EXR" not in matched_null


def test_advice_renders_with_no_leftover_braces(real_sops: list, benign_facts: dict):
    """Verify that when any SOP matches, all {placeholders} are fully rendered with no unreplaced braces."""
    # Create conditions where multiple SOPs match with real numbers
    facts = {
        **benign_facts,
        "uv_index_max": 9.2,
        "apparent_temperature": 41.5,
        "wind_gusts": 48.0,
        "precipitation_probability_max": 85.0,
        "precipitation_sum": 12.4,
        "temperature": 35.0,
        "weather_code": [95],
        "pressure_msl": 998.0,
    }
    all_tags = [
        "running", "cycling", "hiking", "outdoor_exercise",
        "two_wheeler", "commute", "driving", "travel",
        "children", "elderly", "pets", "picnic", "outdoor_event",
    ]
    results, matched_ids, _ = evaluate_all(real_sops, facts, all_tags)

    leftover_pattern = re.compile(r"\{[a-zA-Z0-9_]+\}")
    for res in results:
        if res.status == "matched":
            assert leftover_pattern.search(res.rendered_advice) is None, (
                f"SOP '{res.sop_id}' has unreplaced placeholder in advice: {res.rendered_advice}"
            )
