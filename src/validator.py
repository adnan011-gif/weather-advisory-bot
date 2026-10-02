"""Validator and deterministic fallback formatting module.

Enforces zero-hallucination guarantees:
- Verifies all numbers in LLM reply exist in the allow-list (facts, thresholds, window hours).
- Verifies all referenced SOP IDs exist in the loaded policy registry.
- Provides templated deterministic answers and code-generated citation footers.
"""

from __future__ import annotations

import datetime
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.sop_engine import SOPResult


SOP_ID_PATTERN = re.compile(r"\b(?:[A-Z]{3,4}-\d{3}|CLR-[A-Z]{3,4})\b")

FACT_LABELS: Dict[str, str] = {
    "wind_gusts": "max wind gusts",
    "wind_speed": "max wind speed",
    "precipitation_sum": "rain",
    "precipitation_probability_max": "rain chance",
    "uv_index_max": "max UV",
    "apparent_temperature": "feels like",
    "temperature": "temp",
    "humidity": "humidity",
    "pressure_msl": "pressure",
}

FACT_UNITS: Dict[str, str] = {
    "wind_gusts": "km/h",
    "wind_speed": "km/h",
    "precipitation_sum": "mm",
    "precipitation_probability_max": "%",
    "uv_index_max": "",
    "apparent_temperature": "°C",
    "temperature": "°C",
    "humidity": "%",
    "pressure_msl": "hPa",
}

PREFERRED_FACT_ORDER = [
    "wind_gusts",
    "precipitation_sum",
    "uv_index_max",
    "apparent_temperature",
    "temperature",
    "wind_speed",
    "precipitation_probability_max",
    "humidity",
    "pressure_msl",
]


def format_conditions_line(
    window_label: str,
    facts_used: Optional[Dict[str, Any]] = None,
    all_facts: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Build deterministic 'Conditions' line from facts used (or key facts)."""
    source_facts = dict(facts_used or {})
    if not source_facts and all_facts:
        for k in ["wind_gusts", "precipitation_sum", "uv_index_max"]:
            if k in all_facts and all_facts[k] is not None:
                source_facts[k] = all_facts[k]

    if not source_facts:
        return None

    ordered_keys = sorted(
        source_facts.keys(),
        key=lambda k: PREFERRED_FACT_ORDER.index(k) if k in PREFERRED_FACT_ORDER else 99,
    )

    parts: List[str] = []
    for k in ordered_keys:
        v = source_facts[k]
        if v is None:
            continue
        if isinstance(v, (list, set, tuple)):
            continue

        label = FACT_LABELS.get(k, k.replace("_", " "))
        unit = FACT_UNITS.get(k, "")

        if isinstance(v, float):
            val_str = f"{v:.1f}"
        else:
            val_str = str(v)

        unit_str = f" {unit}" if unit and unit != "%" else unit
        parts.append(f"{label} {val_str}{unit_str}".strip())

    if not parts:
        return None

    return f"Conditions ({window_label}): {', '.join(parts)}"


def format_local_fetch_time(
    fetch_time_raw: str,
    timezone_name: Optional[str] = None,
    utc_offset_seconds: Optional[int] = None,
) -> str:
    """Format fetch time as local time with timezone abbreviation, no microseconds."""
    try:
        clean_str = fetch_time_raw.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(clean_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)

        tz = None
        if timezone_name:
            try:
                from zoneinfo import ZoneInfo
                tz = ZoneInfo(timezone_name)
            except Exception:
                pass
        if tz is None and utc_offset_seconds is not None:
            tz = datetime.timezone(datetime.timedelta(seconds=utc_offset_seconds))

        if tz is not None:
            dt = dt.astimezone(tz)

        tz_abbr = dt.strftime("%Z") or "UTC"
        dt_str = dt.strftime("%Y-%m-%d %H:%M")
        return f"{dt_str} {tz_abbr}".strip()
    except Exception:
        if "." in fetch_time_raw:
            return fetch_time_raw.split(".")[0]
        return fetch_time_raw


def validate_reply(
    reply: str,
    allowed_numbers: Set[float | int],
    known_sop_ids: Set[str],
) -> Tuple[bool, Optional[str]]:
    """Validate that LLM reply contains only allowed numbers and recognized SOP IDs.

    Args:
        reply: Unvalidated text returned by LLM composer.
        allowed_numbers: Allow-list of all numbers from facts, thresholds, window hours, etc.
        known_sop_ids: Set of valid SOP IDs currently loaded.

    Returns:
        (is_valid, error_reason)
    """
    if not reply:
        return False, "Reply is empty."

    # 1. Check referenced SOP IDs
    found_ids = SOP_ID_PATTERN.findall(reply)
    for sop_id in found_ids:
        if sop_id not in known_sop_ids:
            return False, f"Cited unknown SOP ID: '{sop_id}'."

    # 2. Strip SOP IDs so digits in IDs (e.g. 001 in EXR-001) are not parsed as standalone numbers
    text_without_ids = SOP_ID_PATTERN.sub(" ", reply)

    # 3. Extract all numeric values
    raw_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text_without_ids)

    # Normalize allowed numbers as floats for comparison
    allowed_floats = {float(n) for n in allowed_numbers}

    for num_str in raw_numbers:
        val = float(num_str)
        # Check if val matches any allowed number within small floating-point tolerance
        matched = any(abs(val - a) < 0.05 for a in allowed_floats)
        if not matched:
            return False, f"Unallowed number in reply: {num_str}."

    return True, None


def build_footer(
    primary: Optional[SOPResult],
    also_applies: List[SOPResult],
    resolved_name: str,
    fetch_time: str,
    skipped_ids: Optional[List[str]] = None,
    conditions_line: Optional[str] = None,
    timezone_name: Optional[str] = None,
    utc_offset_seconds: Optional[int] = None,
) -> str:
    """Build deterministic citation footer appended by code."""
    lines = ["\n\n---"]

    if primary:
        lines.append(f"Policy: {primary.sop_id} ({primary.effective_severity})")

    # Filter out clear SOPs from also_applies just in case
    filtered_also = [r for r in also_applies if r.match_type != "clear"]
    if filtered_also:
        also_str = ", ".join(f"{r.sop_id} ({r.effective_severity})" for r in filtered_also)
        lines.append(f"Also applies: {also_str}")

    if conditions_line:
        lines.append(conditions_line)

    if skipped_ids:
        lines.append(
            f"Note: {len(skipped_ids)} safety checks could not be run (missing data): {', '.join(skipped_ids)}"
        )

    lines.append(f"Location: {resolved_name}")
    formatted_fetch = format_local_fetch_time(
        fetch_time,
        timezone_name=timezone_name,
        utc_offset_seconds=utc_offset_seconds,
    )
    lines.append(f"Data fetched: {formatted_fetch}")

    return "\n".join(lines)


def templated_answer(
    primary: Optional[SOPResult],
    also_applies: List[SOPResult],
    resolved_name: str,
    time_ref: str,
    fetch_time: str,
    skipped_ids: Optional[List[str]] = None,
    window_note: Optional[str] = None,
    assumed_time: bool = False,
    previous_change_note: Optional[str] = None,
    conditions_line: Optional[str] = None,
    timezone_name: Optional[str] = None,
    utc_offset_seconds: Optional[int] = None,
) -> str:
    """Generate a fully deterministic advisory response without calling the LLM.

    Used when LLM composer fails or produces an unallowed number / unknown ID.
    """
    body_parts = []

    # State assumption if time_ref was not explicitly provided by user
    if assumed_time:
        body_parts.append(f"Advisory for {resolved_name} (assuming time window '{time_ref}'):")
    else:
        body_parts.append(f"Advisory for {resolved_name} ({time_ref}):")

    if previous_change_note:
        body_parts.append(f"[{previous_change_note}]")

    if window_note:
        body_parts.append(f"({window_note})")

    if primary:
        body_parts.append(primary.rendered_advice)

    filtered_also = [r for r in also_applies if r.match_type != "clear"]
    if filtered_also:
        body_parts.append("Additional applicable guidelines:")
        for r in filtered_also:
            if r.rendered_advice:
                body_parts.append(f"- {r.rendered_advice}")

    if skipped_ids:
        body_parts.append(
            f"{len(skipped_ids)} safety checks could not be run (missing data): {', '.join(skipped_ids)}."
        )

    main_text = "\n\n".join(body_parts)
    footer = build_footer(
        primary=primary,
        also_applies=also_applies,
        resolved_name=resolved_name,
        fetch_time=fetch_time,
        skipped_ids=skipped_ids,
        conditions_line=conditions_line,
        timezone_name=timezone_name,
        utc_offset_seconds=utc_offset_seconds,
    )

    return main_text + footer
