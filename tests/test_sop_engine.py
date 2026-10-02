"""Comprehensive unit tests for SOP schema, loader, engine, and conflict resolution.

All tests utilize temporary SOP directories via pytest tmp_path fixtures.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import yaml

from src.facts_registry import FactsRegistry
from src.sop_loader import load_sops, derive_vocabulary, SOPLoadError
from src.sop_engine import evaluate_all, evaluate_sop, extract_thresholds
from src.conflict_resolver import resolve


@pytest.fixture
def facts_registry() -> FactsRegistry:
    """Fixture providing initialized FactsRegistry."""
    return FactsRegistry()


def test_threshold_boundary(tmp_path: Path, facts_registry: FactsRegistry):
    """Test strict boundary semantics for > vs >=."""
    sop_gt = {
        "id": "TEST-GT",
        "title": "Greater Than Test",
        "category": "heat",
        "severity": "high",
        "priority": 1,
        "applies_to": ["running"],
        "match_type": "numeric",
        "condition": {"fact": "temperature", "op": ">", "value": 30.0},
        "advice": "High heat warning: {temperature}°C.",
    }
    sop_gte = {
        "id": "TEST-GTE",
        "title": "Greater Than Equal Test",
        "category": "heat",
        "severity": "high",
        "priority": 1,
        "applies_to": ["running"],
        "match_type": "numeric",
        "condition": {"fact": "temperature", "op": ">=", "value": 30.0},
        "advice": "High heat warning: {temperature}°C.",
    }

    with open(tmp_path / "gt.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_gt, f)
    with open(tmp_path / "gte.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_gte, f)

    sops = load_sops(tmp_path, facts_registry)
    sops_map = {s.id: s for s in sops}

    # At exactly 30.0: GT is not matched, GTE is matched
    res_gt_30 = evaluate_sop(sops_map["TEST-GT"], {"temperature": 30.0}, ["running"])
    assert res_gt_30.status == "not_matched"

    res_gte_30 = evaluate_sop(sops_map["TEST-GTE"], {"temperature": 30.0}, ["running"])
    assert res_gte_30.status == "matched"
    assert "30.0°C" in res_gte_30.rendered_advice

    # At 30.1: Both matched
    res_gt_301 = evaluate_sop(sops_map["TEST-GT"], {"temperature": 30.1}, ["running"])
    assert res_gt_301.status == "matched"


def test_missing_fact_is_unevaluable(tmp_path: Path, facts_registry: FactsRegistry):
    """Test that missing or None fact produces status 'unevaluable', not false."""
    sop_data = {
        "id": "TEST-UNEVAL",
        "title": "Unevaluable Test",
        "category": "wind",
        "severity": "moderate",
        "priority": 1,
        "applies_to": ["cycling"],
        "match_type": "numeric",
        "condition": {"fact": "wind_speed", "op": ">=", "value": 25.0},
        "advice": "Wind advisory.",
    }
    with open(tmp_path / "test.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_data, f)

    sops = load_sops(tmp_path, facts_registry)
    sop = sops[0]

    # Explicit None
    res_none = evaluate_sop(sop, {"wind_speed": None}, ["cycling"])
    assert res_none.status == "unevaluable"

    # Missing from dict entirely
    res_missing = evaluate_sop(sop, {}, ["cycling"])
    assert res_missing.status == "unevaluable"


def test_any_of_with_one_true_and_one_unevaluable(tmp_path: Path, facts_registry: FactsRegistry):
    """Test any_of: a single True condition matches even if another is unevaluable."""
    sop_data = {
        "id": "TEST-ANY",
        "title": "Any Of Logic Test",
        "category": "composite",
        "severity": "high",
        "priority": 1,
        "applies_to": ["hiking"],
        "match_type": "composite",
        "condition": {
            "any_of": [
                {"fact": "wind_speed", "op": ">=", "value": 40.0},
                {"fact": "precipitation_sum", "op": ">=", "value": 10.0},
            ]
        },
        "advice": "Adverse conditions.",
    }
    with open(tmp_path / "test.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_data, f)

    sops = load_sops(tmp_path, facts_registry)
    sop = sops[0]

    # wind_speed is True (45 >= 40), precipitation_sum is None -> MATCHED
    res = evaluate_sop(sop, {"wind_speed": 45.0, "precipitation_sum": None}, ["hiking"])
    assert res.status == "matched"

    # wind_speed is False (20 < 40), precipitation_sum is None -> UNEVALUABLE
    res_uneval = evaluate_sop(sop, {"wind_speed": 20.0, "precipitation_sum": None}, ["hiking"])
    assert res_uneval.status == "unevaluable"


def test_all_of_with_one_false_and_one_unevaluable(tmp_path: Path, facts_registry: FactsRegistry):
    """Test all_of: a single False condition renders it not_matched even if another is unevaluable."""
    sop_data = {
        "id": "TEST-ALL",
        "title": "All Of Logic Test",
        "category": "composite",
        "severity": "high",
        "priority": 1,
        "applies_to": ["hiking"],
        "match_type": "composite",
        "condition": {
            "all_of": [
                {"fact": "wind_speed", "op": ">=", "value": 40.0},
                {"fact": "precipitation_sum", "op": ">=", "value": 10.0},
            ]
        },
        "advice": "Severe storm conditions.",
    }
    with open(tmp_path / "test.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_data, f)

    sops = load_sops(tmp_path, facts_registry)
    sop = sops[0]

    # wind_speed is False (15 < 40), precipitation_sum is None -> NOT_MATCHED
    res = evaluate_sop(sop, {"wind_speed": 15.0, "precipitation_sum": None}, ["hiking"])
    assert res.status == "not_matched"

    # wind_speed is True (50 >= 40), precipitation_sum is None -> UNEVALUABLE
    res_uneval = evaluate_sop(sop, {"wind_speed": 50.0, "precipitation_sum": None}, ["hiking"])
    assert res_uneval.status == "unevaluable"


def test_semantic_tier_ordering(tmp_path: Path, facts_registry: FactsRegistry):
    """Test that semantic rubrics evaluate tiers in order and select the first winning tier."""
    sop_data = {
        "id": "SEM-001",
        "title": "Wind Comfort Tiers",
        "category": "wind",
        "severity": "moderate",
        "priority": 1,
        "applies_to": ["cycling"],
        "match_type": "semantic",
        "intent_description": "Wind advisory based on rider comfort levels.",
        "rubric": [
            {
                "name": "tier_danger",
                "condition": {"fact": "wind_speed", "op": ">=", "value": 45.0},
                "severity": "critical",
                "advice": "Dangerous gale winds of {wind_speed} km/h.",
            },
            {
                "name": "tier_caution",
                "condition": {"fact": "wind_speed", "op": ">=", "value": 25.0},
                "severity": "moderate",
                "advice": "Breezy conditions of {wind_speed} km/h.",
            },
        ],
    }
    with open(tmp_path / "sem.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_data, f)

    sops = load_sops(tmp_path, facts_registry)
    sop = sops[0]

    # Wind = 50: hits tier_danger
    res_danger = evaluate_sop(sop, {"wind_speed": 50.0}, ["cycling"])
    assert res_danger.status == "matched"
    assert res_danger.tier == "tier_danger"
    assert res_danger.effective_severity == "critical"
    assert "Dangerous gale winds of 50.0 km/h." in res_danger.rendered_advice

    # Wind = 30: skips tier_danger, hits tier_caution
    res_caution = evaluate_sop(sop, {"wind_speed": 30.0}, ["cycling"])
    assert res_caution.status == "matched"
    assert res_caution.tier == "tier_caution"
    assert res_caution.effective_severity == "moderate"
    assert "Breezy conditions of 30.0 km/h." in res_caution.rendered_advice

    # Wind = 10: matches no tier
    res_none = evaluate_sop(sop, {"wind_speed": 10.0}, ["cycling"])
    assert res_none.status == "not_matched"


def test_clear_sop_fires_and_blocked_when_hazard_unevaluable(tmp_path: Path, facts_registry: FactsRegistry):
    """Test that clear SOP fires when all hazards are false, but becomes unevaluable if any hazard is unevaluable."""
    hazard_sop = {
        "id": "HAZ-001",
        "title": "High Heat",
        "category": "heat",
        "severity": "high",
        "priority": 1,
        "applies_to": ["hiking"],
        "match_type": "numeric",
        "condition": {"fact": "temperature", "op": ">=", "value": 35.0},
        "advice": "Stay hydrated.",
    }
    clear_sop = {
        "id": "CLR-001",
        "title": "Comfortable Hiking Weather",
        "category": "heat",
        "severity": "info",
        "priority": 10,
        "applies_to": ["hiking"],
        "match_type": "clear",
        "advice": "Conditions are safe and clear for hiking.",
    }

    with open(tmp_path / "haz.yaml", "w", encoding="utf-8") as f:
        yaml.dump(hazard_sop, f)
    with open(tmp_path / "clr.yaml", "w", encoding="utf-8") as f:
        yaml.dump(clear_sop, f)

    sops = load_sops(tmp_path, facts_registry)

    # 1. Temperature is safe (22°C): Clear SOP fires
    results, matched, uneval = evaluate_all(sops, {"temperature": 22.0}, ["hiking"])
    assert "CLR-001" in matched
    assert "HAZ-001" not in matched

    # 2. Temperature is hazardous (38°C): Hazard matches, Clear SOP does not match
    results, matched, uneval = evaluate_all(sops, {"temperature": 38.0}, ["hiking"])
    assert "HAZ-001" in matched
    assert "CLR-001" not in matched

    # 3. Temperature is missing (None): Hazard is unevaluable, Clear SOP MUST NOT fire
    results, matched, uneval = evaluate_all(sops, {"temperature": None}, ["hiking"])
    assert "HAZ-001" in uneval
    assert "CLR-001" in uneval
    assert len(matched) == 0


def test_override_beats_critical_non_override(tmp_path: Path, facts_registry: FactsRegistry):
    """Test that an SOP with override=True beats a critical severity SOP with override=False."""
    sop_critical = {
        "id": "CRIT-001",
        "title": "Severe Heat",
        "category": "heat",
        "severity": "critical",
        "priority": 1,
        "override": False,
        "applies_to": ["running"],
        "match_type": "numeric",
        "condition": {"fact": "temperature", "op": ">=", "value": 35.0},
        "advice": "Danger.",
    }
    sop_override = {
        "id": "OVR-001",
        "title": "Special Event Override",
        "category": "events",
        "severity": "low",
        "priority": 5,
        "override": True,
        "applies_to": ["running"],
        "match_type": "numeric",
        "condition": {"fact": "temperature", "op": ">=", "value": 20.0},
        "advice": "Special event active.",
    }

    with open(tmp_path / "crit.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_critical, f)
    with open(tmp_path / "ovr.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_override, f)

    sops = load_sops(tmp_path, facts_registry)
    results, matched, _ = evaluate_all(sops, {"temperature": 38.0}, ["running"])
    matched_results = [r for r in results if r.status == "matched"]

    resolution = resolve(matched_results)
    assert resolution.primary is not None
    assert resolution.primary.sop_id == "OVR-001"
    assert "override" in resolution.reason


def test_tie_breaking_by_priority_then_id(tmp_path: Path, facts_registry: FactsRegistry):
    """Test conflict resolution tie breaking: priority first, then id."""
    sop_p1 = {
        "id": "SOP-B",
        "title": "Rule B",
        "category": "general",
        "severity": "high",
        "priority": 1,
        "applies_to": ["any"],
        "match_type": "numeric",
        "condition": {"fact": "wind_speed", "op": ">=", "value": 10.0},
        "advice": "Rule B.",
    }
    sop_p2 = {
        "id": "SOP-A",
        "title": "Rule A",
        "category": "general",
        "severity": "high",
        "priority": 2,
        "applies_to": ["any"],
        "match_type": "numeric",
        "condition": {"fact": "wind_speed", "op": ">=", "value": 10.0},
        "advice": "Rule A.",
    }

    with open(tmp_path / "p1.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_p1, f)
    with open(tmp_path / "p2.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_p2, f)

    sops = load_sops(tmp_path, facts_registry)
    results, _, _ = evaluate_all(sops, {"wind_speed": 15.0}, ["any"])
    matched = [r for r in results if r.status == "matched"]

    res = resolve(matched)
    # SOP-B wins because priority 1 < 2
    assert res.primary.sop_id == "SOP-B"
    assert "priority" in res.reason

    # Now tie priority: SOP-A beats SOP-B by ID
    sop_p1["priority"] = 2
    with open(tmp_path / "p1.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_p1, f)
    sops_tied = load_sops(tmp_path, facts_registry)
    results_tied, _, _ = evaluate_all(sops_tied, {"wind_speed": 15.0}, ["any"])
    res_id = resolve([r for r in results_tied if r.status == "matched"])
    assert res_id.primary.sop_id == "SOP-A"
    assert "tie-breaking on ID" in res_id.reason


def test_intersects_operator_on_weather_codes(tmp_path: Path, facts_registry: FactsRegistry):
    """Test intersects operator evaluating set-valued weather_code against target list."""
    sop_data = {
        "id": "THUN-001",
        "title": "Thunderstorm Warning",
        "category": "storm",
        "severity": "critical",
        "priority": 1,
        "applies_to": ["any"],
        "match_type": "numeric",
        "condition": {
            "fact": "weather_code",
            "op": "intersects",
            "value": [95, 96, 99],
        },
        "advice": "Lightning danger.",
    }
    with open(tmp_path / "thun.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_data, f)

    sops = load_sops(tmp_path, facts_registry)
    sop = sops[0]

    # Matching set
    assert evaluate_sop(sop, {"weather_code": [0, 1, 95]}, ["hiking"]).status == "matched"
    # Non-matching set
    assert evaluate_sop(sop, {"weather_code": [0, 1, 2]}, ["hiking"]).status == "not_matched"


def test_malformed_yaml_error_names_file(tmp_path: Path, facts_registry: FactsRegistry):
    """Verify that a malformed YAML file raises an error naming the file."""
    bad_file = tmp_path / "broken_syntax.yaml"
    bad_file.write_text("id: BROKEN\ncategory: [unclosed list", encoding="utf-8")

    with pytest.raises(SOPLoadError) as exc_info:
        load_sops(tmp_path, facts_registry)
    assert "broken_syntax.yaml" in str(exc_info.value)


def test_unknown_fact_in_condition_rejected_at_load(tmp_path: Path, facts_registry: FactsRegistry):
    """Verify that referencing a fact not in facts.yaml raises an error naming the field."""
    sop_data = {
        "id": "BAD-FACT",
        "title": "Invalid Fact Reference",
        "category": "general",
        "severity": "low",
        "priority": 1,
        "applies_to": ["any"],
        "match_type": "numeric",
        "condition": {"fact": "unknown_solar_metric", "op": ">=", "value": 5},
        "advice": "Test.",
    }
    with open(tmp_path / "bad_fact.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_data, f)

    with pytest.raises(SOPLoadError) as exc_info:
        load_sops(tmp_path, facts_registry)
    assert "unknown_solar_metric" in str(exc_info.value)
    assert "bad_fact.yaml" in str(exc_info.value)


def test_duplicate_id_rejected(tmp_path: Path, facts_registry: FactsRegistry):
    """Verify that duplicate SOP IDs across different files raise an error."""
    sop_1 = {
        "id": "DUP-001",
        "title": "First",
        "category": "heat",
        "severity": "low",
        "priority": 1,
        "applies_to": ["any"],
        "match_type": "numeric",
        "condition": {"fact": "temperature", "op": ">=", "value": 20.0},
        "advice": "First.",
    }
    sop_2 = {
        "id": "DUP-001",
        "title": "Second",
        "category": "wind",
        "severity": "high",
        "priority": 2,
        "applies_to": ["any"],
        "match_type": "numeric",
        "condition": {"fact": "wind_speed", "op": ">=", "value": 30.0},
        "advice": "Second.",
    }

    with open(tmp_path / "first.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_1, f)
    with open(tmp_path / "second.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_2, f)

    with pytest.raises(SOPLoadError) as exc_info:
        load_sops(tmp_path, facts_registry)
    assert "duplicate SOP ID 'DUP-001'" in str(exc_info.value)


def test_newly_added_yaml_changes_behaviour_no_code_change(tmp_path: Path, facts_registry: FactsRegistry):
    """Verify adding a new YAML file dynamically changes matching behavior with zero code changes."""
    sop_initial = {
        "id": "BASE-001",
        "title": "Normal Breeze",
        "category": "wind",
        "severity": "low",
        "priority": 5,
        "applies_to": ["cycling"],
        "match_type": "numeric",
        "condition": {"fact": "wind_speed", "op": ">=", "value": 15.0},
        "advice": "Mild breeze.",
    }
    with open(tmp_path / "base.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_initial, f)

    # Initial run: BASE-001 is primary
    sops_1 = load_sops(tmp_path, facts_registry)
    res_1, matched_1, _ = evaluate_all(sops_1, {"wind_speed": 40.0}, ["cycling"])
    resolution_1 = resolve([r for r in res_1 if r.status == "matched"])
    assert resolution_1.primary.sop_id == "BASE-001"

    # Dynamically drop a new high-severity emergency SOP into the directory
    sop_new = {
        "id": "GALE-001",
        "title": "Gale Warning",
        "category": "wind",
        "severity": "critical",
        "priority": 1,
        "applies_to": ["cycling"],
        "match_type": "numeric",
        "condition": {"fact": "wind_speed", "op": ">=", "value": 35.0},
        "advice": "Dangerous gales!",
    }
    with open(tmp_path / "gale.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_new, f)

    # Re-evaluate from directory: GALE-001 immediately takes precedence with zero code change
    sops_2 = load_sops(tmp_path, facts_registry)
    res_2, matched_2, _ = evaluate_all(sops_2, {"wind_speed": 40.0}, ["cycling"])
    resolution_2 = resolve([r for r in res_2 if r.status == "matched"])
    assert resolution_2.primary.sop_id == "GALE-001"
    assert len(resolution_2.also_applies) == 1
    assert resolution_2.also_applies[0].sop_id == "BASE-001"


def test_vocabulary_derived_from_sops(tmp_path: Path, facts_registry: FactsRegistry):
    """Verify dynamic activity vocabulary derivation from loaded SOPs."""
    sop_trail = {
        "id": "TRL-001",
        "title": "Trail Running Heat Policy",
        "category": "running",
        "severity": "moderate",
        "priority": 1,
        "applies_to": ["trail_running", "mountain_jogging"],
        "match_type": "numeric",
        "condition": {"fact": "temperature", "op": ">=", "value": 28.0},
        "advice": "Trail heat.",
        "intent_description": "Safety rules and considerations for rugged trail running.",
    }
    sop_kayak = {
        "id": "KYK-001",
        "title": "Whitewater Kayak Wind Policy",
        "category": "water_sports",
        "severity": "high",
        "priority": 1,
        "applies_to": ["kayaking", "any"],
        "match_type": "numeric",
        "condition": {"fact": "wind_speed", "op": ">=", "value": 30.0},
        "advice": "High waves.",
    }

    with open(tmp_path / "trail.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_trail, f)
    with open(tmp_path / "kayak.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_kayak, f)

    sops = load_sops(tmp_path, facts_registry)
    vocab = derive_vocabulary(sops)

    # "any" is excluded
    assert "any" not in vocab

    # tags are present
    assert "trail_running" in vocab
    assert "mountain_jogging" in vocab
    assert "kayaking" in vocab

    # Rich description used
    assert vocab["trail_running"]["title"] == "Trail Running Heat Policy"
    assert "rugged trail running" in vocab["trail_running"]["intent_description"]


def test_thresholds_extraction(tmp_path: Path, facts_registry: FactsRegistry):
    """Test extracting numeric thresholds from conditions and advice text."""
    sop_data = {
        "id": "THRESH-001",
        "title": "Threshold Extractor Test",
        "category": "heat",
        "severity": "high",
        "priority": 1,
        "applies_to": ["running"],
        "match_type": "numeric",
        "condition": {"fact": "temperature", "op": ">=", "value": 32.5},
        "advice": "Take a 15 minute break every 45 minutes when {temperature}°C.",
    }
    with open(tmp_path / "thresh.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sop_data, f)

    sops = load_sops(tmp_path, facts_registry)
    thresholds = extract_thresholds(sops[0])

    assert 32.5 in thresholds
    assert 15 in thresholds
    assert 45 in thresholds
