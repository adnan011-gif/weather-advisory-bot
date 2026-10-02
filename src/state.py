"""LangGraph State definition for Weather Advisory Support Bot."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class WeatherAdvisoryState(TypedDict, total=False):
    """Unified graph state tracking session context, decisions, and turn progression."""
    # Conversation and input
    messages: List[Dict[str, str]]
    query: str

    # Geographical and session context
    location: Optional[str]
    resolved_name: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    timezone: Optional[str]
    utc_offset_seconds: Optional[int]

    # Intent and activity
    activity_tags: List[str]
    time_ref: Optional[str]
    assumed_time: bool

    # Weather facts and retrieval
    facts: Dict[str, Any]
    raw_fetch_time: Optional[str]
    window_note: Optional[str]

    # SOP Evaluation and Conflict Resolution
    results: List[Dict[str, Any]]
    matched: List[str]
    unevaluable_ids: List[str]
    primary: Optional[Dict[str, Any]]
    also_applies: List[Dict[str, Any]]
    previous_change_note: Optional[str]

    # Audit log and session memory
    decision_log: List[Dict[str, Any]]

    # Response generation
    reply: Optional[str]
    kind: Optional[str]
    error_message: Optional[str]
    error_reason: Optional[str]
    model_used: Optional[str]

    # Internal turn parsing fields
    _parsed_intent: Optional[str]
    _parsed_location: Optional[str]
    _parsed_tags: List[str]
    _parsed_time: Optional[str]
    _cited_sop_ids: List[str]
    _raw_weather: Any
    _raw_composed: Optional[str]
