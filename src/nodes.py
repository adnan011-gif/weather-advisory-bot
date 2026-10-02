"""Graph node definitions for Weather Advisory Support Bot.

Distinguishes deterministic nodes from LLM calls.
All business logic, fallback branching, state persistence, and verification are handled here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from src.state import WeatherAdvisoryState
from src.facts_registry import FactsRegistry
from src.weather_client import (
    WeatherClientProtocol,
    LocationError,
    WeatherAPIError,
    WindowPassedError,
    sanitize_location,
)
from src.sop_schema import SOP
from src.sop_loader import load_sops, derive_vocabulary, SOPLoadError
from src.sop_engine import evaluate_all, SOPResult, extract_thresholds
from src.conflict_resolver import resolve
from src.llm_client import LLMClientProtocol, LLMError
from src.llm_tasks import parse_intent, compose_answer, ParseFailed
from src.validator import validate_reply, build_footer, templated_answer, format_conditions_line


class GraphNodes:
    """Encapsulates graph nodes with injected clients and configurations."""

    def __init__(
        self,
        weather_client: WeatherClientProtocol,
        llm: LLMClientProtocol,
        sops: List[SOP],
        facts_registry: FactsRegistry,
        vocabulary: Dict[str, Dict[str, str]],
        sop_dir: Optional[str | Path] = None,
        facts_path: Optional[str | Path] = None,
    ) -> None:
        self.weather_client = weather_client
        self.llm = llm
        base_dir = Path(__file__).resolve().parent.parent
        self.sop_dir = Path(sop_dir) if sop_dir else (base_dir / "sops")
        self.facts_path = Path(facts_path) if facts_path else (base_dir / "config" / "facts.yaml")
        self.sops = sops
        self.sops_dict: Dict[str, SOP] = {s.id: s for s in sops}
        self.facts_registry = facts_registry
        self.vocabulary = vocabulary

    def reload_policies(self) -> None:
        """Reload facts.yaml and all SOP YAML files from disk at the start of every turn."""
        self.facts_registry = FactsRegistry(config_path=self.facts_path)
        if hasattr(self.weather_client, "facts_registry"):
            self.weather_client.facts_registry = self.facts_registry
        self.sops = load_sops(self.sop_dir, facts_registry=self.facts_registry)
        self.sops_dict = {s.id: s for s in self.sops}
        self.vocabulary = derive_vocabulary(self.sops)

    # 1. Parse Intent (LLM)
    def parse_intent_node(self, state: WeatherAdvisoryState) -> Dict[str, Any]:
        """Classify user query and extract structured intent."""
        query = state.get("query", "")
        # Append user query to message history (capped at last 10 messages)
        msgs = list(state.get("messages", []))
        msgs.append({"role": "user", "content": query})
        msgs = msgs[-10:]

        # SOP Freshness: reload SOPs and facts.yaml from disk at start of EVERY turn
        try:
            self.reload_policies()
        except (SOPLoadError, ValueError, FileNotFoundError) as err:
            return {
                "messages": msgs,
                "kind": "format_error",
                "reply": f"Policy configuration error: {err}",
                "error_message": str(err),
            }

        try:
            parsed = parse_intent(self.llm, query, self.vocabulary)
            model_used = getattr(parsed, "model_used", None) or getattr(self.llm, "model_used", None)
            return {
                "messages": msgs,
                "kind": None,
                "error_message": None,
                "error_reason": None,
                "model_used": model_used,
                "_parsed_intent": parsed.intent,
                "_parsed_location": parsed.location,
                "_parsed_tags": parsed.activity_tags,
                "_parsed_time": parsed.time_ref,
                "_cited_sop_ids": parsed.cited_sop_ids,
            }
        except ParseFailed as err:
            reason = getattr(err, "reason", "invalid_json")
            model_used = getattr(self.llm, "model_used", None)

            if reason == "safety_blocked":
                kind = "request_blocked"
            elif reason in ("rate_limited", "empty_response") or reason.startswith("api_error"):
                kind = "llm_unavailable"
            else:
                kind = "parse_failed"

            decision_log = list(state.get("decision_log", []))
            decision_log.append({
                "turn": len(decision_log) + 1,
                "kind": kind,
                "error_reason": reason,
                "model_used": model_used,
            })
            return {
                "messages": msgs,
                "kind": kind,
                "error_reason": reason,
                "model_used": model_used,
                "decision_log": decision_log,
            }

    # 2. Intent Routing Helper (Deterministic)
    @staticmethod
    def route_intent(state: WeatherAdvisoryState) -> str:
        """Route to appropriate graph branch based on parsed intent or error outcome."""
        kind = state.get("kind")
        if kind == "parse_failed":
            return "parse_failed"
        elif kind == "llm_unavailable":
            return "llm_unavailable"
        elif kind == "request_blocked":
            return "request_blocked"

        intent = state.get("_parsed_intent")
        if intent == "out_of_scope":
            return "no_guidance"
        elif intent == "explain":
            return "explain_decision"
        elif intent == "advice":
            return "resolve_context"
        return "no_guidance"

    def parse_failed_node(self, state: WeatherAdvisoryState) -> Dict[str, Any]:
        """User message could not be parsed into valid schema."""
        return {
            "kind": "parse_failed",
            "reply": "I couldn't understand your request, please rephrase.",
        }

    def llm_unavailable_node(self, state: WeatherAdvisoryState) -> Dict[str, Any]:
        """Language service unavailable (rate limit, daily quota, or API failure)."""
        return {
            "kind": "llm_unavailable",
            "reply": "My language service is temporarily unavailable (usage limit reached), so I can't answer right now. Please try again later.",
        }

    def request_blocked_node(self, state: WeatherAdvisoryState) -> Dict[str, Any]:
        """Request was blocked by safety filters."""
        return {
            "kind": "request_blocked",
            "reply": "I couldn't process that request.",
        }

    def no_guidance_node(self, state: WeatherAdvisoryState) -> Dict[str, Any]:
        """Polite guidance message when query is out of scope or unsupported."""
        return {
            "kind": "no_guidance",
            "reply": "I don't have safety guidance for that activity or query. I specialize in outdoor activity safety based on live weather data.",
        }

    # 3. Explain Decision (Deterministic)
    def explain_decision_node(self, state: WeatherAdvisoryState) -> Dict[str, Any]:
        """Explain the rationale behind a past advisory from the decision log."""
        decision_log = state.get("decision_log", [])
        if not decision_log:
            return {
                "kind": "explain",
                "reply": "There is nothing to explain yet. Ask for a weather safety advisory first.",
            }

        cited_ids = state.get("_cited_sop_ids", [])
        # Check if user cited an SOP ID that does not exist in registry
        for cid in cited_ids:
            if cid not in self.sops_dict:
                return {
                    "kind": "explain",
                    "reply": f"SOP '{cid}' does not exist in our policy registry.",
                }

        # Explain from the most recent decision log entry
        last_turn = decision_log[-1]
        primary_id = last_turn.get("primary_id")
        primary_sop = self.sops_dict.get(primary_id)
        primary_title = primary_sop.title if primary_sop else primary_id

        facts_used_str = ", ".join(
            f"{k}={v}" for k, v in last_turn.get("facts_used", {}).items() if v is not None
        )
        if not facts_used_str:
            facts_used_str = "no hazard thresholds reached"

        lines = [
            f"Explanation for {last_turn.get('resolved_name', 'your location')} ({last_turn.get('time_ref', 'today')}):",
            f"- Primary Advisory: {primary_id} — '{primary_title}' ({last_turn.get('severity')}).",
            f"- Justification: {last_turn.get('reason')}",
            f"- Meteorological Facts Checked: {facts_used_str}.",
        ]

        skipped = last_turn.get("skipped_ids", [])
        if skipped:
            lines.append(f"- Note: {len(skipped)} safety checks could not be run due to missing data: {', '.join(skipped)}.")

        return {
            "kind": "explain",
            "reply": "\n".join(lines),
        }

    # 4. Resolve Context & Session State (Deterministic)
    def resolve_context_node(self, state: WeatherAdvisoryState) -> Dict[str, Any]:
        """Merge turn inputs with thread history, apply time defaults, sanitize inputs."""
        updates: Dict[str, Any] = {}

        # Merge Location: new location replaces old; message without location reuses saved
        parsed_loc = state.get("_parsed_location")
        current_loc = state.get("location")
        if parsed_loc:
            try:
                clean_loc = sanitize_location(parsed_loc)
                if clean_loc != current_loc:
                    updates["location"] = clean_loc
                    updates["resolved_name"] = None  # Force fresh geocoding
                    updates["lat"] = None
                    updates["lon"] = None
            except LocationError:
                updates["location"] = None
        elif current_loc:
            updates["location"] = current_loc

        # Merge Activity Tags: new tags replace old; message without tags reuses saved
        parsed_tags = state.get("_parsed_tags", [])
        current_tags = state.get("activity_tags", [])
        if parsed_tags:
            updates["activity_tags"] = parsed_tags
        elif current_tags:
            updates["activity_tags"] = current_tags
        else:
            updates["activity_tags"] = []

        # Merge Time Reference: explicit beats saved; no time -> default 'today' with assumed_time=True
        parsed_time = state.get("_parsed_time")
        current_time = state.get("time_ref")
        if parsed_time:
            updates["time_ref"] = parsed_time
            updates["assumed_time"] = False
        elif current_time:
            updates["time_ref"] = current_time
            updates["assumed_time"] = state.get("assumed_time", False)
        else:
            updates["time_ref"] = "today"
            updates["assumed_time"] = True

        # Check for missing location or missing activity
        eff_loc = updates.get("location", state.get("location"))
        eff_tags = updates.get("activity_tags", state.get("activity_tags"))

        if not eff_loc:
            updates["kind"] = "ask_location"
            updates["reply"] = "Please specify the city or location you plan to visit."
        elif not eff_tags:
            updates["kind"] = "ask_activity"
            updates["reply"] = "Please specify the outdoor activity you are planning (e.g. running, cycling, hiking, picnic, travel)."

        return updates

    # 5. Geocode (Deterministic)
    def geocode_node(self, state: WeatherAdvisoryState) -> Dict[str, Any]:
        """Resolve location string to coordinates and formatted place name."""
        if state.get("resolved_name") and state.get("lat") is not None:
            return {}  # Location already resolved

        loc = state.get("location", "")
        try:
            res = self.weather_client.geocode(loc)
            return {
                "lat": res.lat,
                "lon": res.lon,
                "resolved_name": res.resolved_name,
                "timezone": res.timezone,
            }
        except LocationError as err:
            return {
                "kind": "format_error",
                "error_message": str(err),
                "reply": f"Could not find coordinates for location '{loc}'. Please check spelling or specify a nearby city.",
            }

    # 6. Fetch Weather (Deterministic)
    def fetch_weather_node(self, state: WeatherAdvisoryState) -> Dict[str, Any]:
        """Fetch fresh live weather forecast for target coordinates."""
        lat = state.get("lat")
        lon = state.get("lon")
        if lat is None or lon is None:
            return {
                "kind": "format_error",
                "reply": "Geographic coordinates were not available for forecast retrieval.",
            }

        try:
            raw = self.weather_client.fetch_weather(lat, lon)
            return {
                "_raw_weather": raw,
                "raw_fetch_time": raw.fetch_time,
            }
        except WeatherAPIError as err:
            return {
                "kind": "format_error",
                "error_message": str(err),
                "reply": f"Weather forecast service is temporarily unavailable: {err}",
            }

    # 7. Compute Facts (Deterministic)
    def compute_facts_node(self, state: WeatherAdvisoryState) -> Dict[str, Any]:
        """Compute aggregated domain facts for the requested window."""
        raw = state.get("_raw_weather")
        time_ref = state.get("time_ref", "today")

        try:
            computed = self.weather_client.compute_facts(raw, window_name=time_ref)
            return {
                "facts": computed.facts,
                "utc_offset_seconds": computed.utc_offset_seconds,
                "window_note": computed.window_note,
                "window_start": computed.window_start,
                "window_end": computed.window_end,
            }
        except WindowPassedError as err:
            return {
                "kind": "window_passed",
                "reply": str(err),
            }

    # 8. Match SOPs (Deterministic)
    def match_sops_node(self, state: WeatherAdvisoryState) -> Dict[str, Any]:
        """Evaluate declarative SOP rules against facts and activity tags."""
        facts = state.get("facts", {})
        tags = state.get("activity_tags", [])
        resolved_name = state.get("resolved_name", "the requested location")

        results, matched_ids, uneval_ids = evaluate_all(self.sops, facts, tags)
        results_dicts = [r.model_dump() for r in results]

        if not matched_ids:
            if uneval_ids:
                return {
                    "results": results_dicts,
                    "matched": [],
                    "unevaluable_ids": uneval_ids,
                    "kind": "data_incomplete",
                    "reply": (
                        f"Live weather data for {resolved_name} was incomplete "
                        f"({len(uneval_ids)} safety checks could not be run: {', '.join(uneval_ids)}), "
                        f"so outdoor safety cannot be reliably confirmed."
                    ),
                }
            return {
                "results": results_dicts,
                "matched": [],
                "unevaluable_ids": [],
                "kind": "no_guidance",
                "reply": f"I don't have safety guidance for that activity or scenario in {resolved_name}.",
            }

        return {
            "results": results_dicts,
            "matched": matched_ids,
            "unevaluable_ids": uneval_ids,
        }

    # 9. Resolve Conflicts (Deterministic)
    def resolve_conflicts_node(self, state: WeatherAdvisoryState) -> Dict[str, Any]:
        """Resolve matched SOPs by precedence rules and update decision log."""
        results_data = state.get("results", [])
        matched_results = [SOPResult(**r) for r in results_data if r.get("status") == "matched"]

        resolution = resolve(matched_results)
        primary = resolution.primary
        also_applies = resolution.also_applies

        # Check if primary changed from previous turn for the same location
        previous_note: Optional[str] = None
        decision_log = list(state.get("decision_log", []))
        if decision_log:
            last_entry = decision_log[-1]
            if last_entry.get("resolved_name") == state.get("resolved_name") and primary:
                prev_id = last_entry.get("primary_id")
                prev_sev = last_entry.get("severity")
                if prev_id and prev_id != primary.sop_id:
                    previous_note = (
                        f"Advisory updated from {prev_id} ({prev_sev}) to {primary.sop_id} ({primary.effective_severity})."
                    )

        # Append turn record to decision log
        turn_entry = {
            "turn": len(decision_log) + 1,
            "location": state.get("location"),
            "resolved_name": state.get("resolved_name"),
            "time_ref": state.get("time_ref"),
            "tags": state.get("activity_tags"),
            "matched_ids": state.get("matched", []),
            "primary_id": primary.sop_id if primary else None,
            "severity": primary.effective_severity if primary else None,
            "reason": resolution.reason,
            "facts_used": primary.facts_used if primary else {},
            "skipped_ids": state.get("unevaluable_ids", []),
            "model_used": state.get("model_used"),
        }
        decision_log.append(turn_entry)

        return {
            "primary": primary.model_dump() if primary else None,
            "also_applies": [r.model_dump() for r in also_applies],
            "previous_change_note": previous_note,
            "decision_log": decision_log,
        }

    # 10. Compose Answer (LLM)
    def compose_node(self, state: WeatherAdvisoryState) -> Dict[str, Any]:
        """Compose language response strictly from structured payload (no raw user message)."""
        primary_data = state.get("primary")
        if not primary_data:
            return {
                "_raw_composed": None,
                "answer_source": "template",
                "_compose_reason": "no_primary",
                "reason": "no_primary",
            }

        primary = SOPResult(**primary_data)

        # Policy: if primary SOP is match_type clear OR effective severity is info,
        # skip the LLM compose call and use templated_answer directly.
        if primary.match_type == "clear" or primary.effective_severity == "info":
            return {
                "_raw_composed": None,
                "answer_source": "template",
                "_compose_reason": "no_llm_needed",
                "reason": "no_llm_needed",
            }

        also_data = state.get("also_applies", [])
        also_applies = [SOPResult(**r) for r in also_data]

        caveat_skipped: Optional[str] = None
        skipped_ids = state.get("unevaluable_ids", [])
        if skipped_ids:
            caveat_skipped = (
                f"{len(skipped_ids)} safety checks could not be run (missing data): {', '.join(skipped_ids)}."
            )

        # Build clean payload containing zero user text.
        # Note: window_note is NOT included; time window note is added strictly by code.
        payload = {
            "activity_tags": state.get("activity_tags", []),
            "time_ref": state.get("time_ref", "today"),
            "resolved_name": state.get("resolved_name", "the requested location"),
            "assumed_time": state.get("assumed_time", False),
            "primary_sop_id": primary.sop_id,
            "primary_sop_severity": primary.effective_severity,
            "primary_sop_advice": primary.rendered_advice,
            "also_applies": [
                {
                    "id": r.sop_id,
                    "title": self.sops_dict[r.sop_id].title if r.sop_id in self.sops_dict else r.sop_id,
                    "severity": r.effective_severity,
                }
                for r in also_applies
            ],
            "facts_used": primary.facts_used,
            "caveat_skipped_checks": caveat_skipped,
            "previous_primary_changed_note": state.get("previous_change_note"),
        }

        try:
            raw_composed = compose_answer(self.llm, payload)
            model_used = getattr(self.llm, "model_used", None) or state.get("model_used")
            return {
                "_raw_composed": raw_composed,
                "answer_source": "llm",
                "_compose_reason": "llm_composed",
                "model_used": model_used,
            }
        except LLMError as err:
            # Fall back to templated answer on LLM exception
            return {
                "_raw_composed": None,
                "answer_source": "template",
                "_compose_reason": f"llm_error:{err.reason}",
                "reason": f"llm_error:{err.reason}",
            }

    # 11. Validate Answer (Deterministic)
    def validate_answer_node(self, state: WeatherAdvisoryState) -> Dict[str, Any]:
        """Verify numbers and SOP IDs in LLM response against allow-list; append code footer."""
        primary_data = state.get("primary")
        primary = SOPResult(**primary_data) if primary_data else None
        also_data = state.get("also_applies", [])
        also_applies = [SOPResult(**r) for r in also_data]

        resolved_name = state.get("resolved_name", "Unknown Location")
        time_ref = state.get("time_ref", "today")
        fetch_time = state.get("raw_fetch_time", "N/A")
        skipped_ids = state.get("unevaluable_ids", [])
        window_note = state.get("window_note")
        assumed_time = state.get("assumed_time", False)
        prev_note = state.get("previous_change_note")
        tz_name = state.get("timezone")
        utc_offset = state.get("utc_offset_seconds")
        model_used = state.get("model_used") or getattr(self.llm, "model_used", None)

        # Determine window label for Conditions line
        win_start = state.get("window_start")
        win_end = state.get("window_end")
        if win_start and win_end:
            window_label = f"{win_start}-{win_end}"
        else:
            window_label = time_ref

        conditions_line = format_conditions_line(
            window_label=window_label,
            facts_used=primary.facts_used if primary else None,
            all_facts=state.get("facts"),
        )

        raw_composed = state.get("_raw_composed")

        def _fallback(reason_text: str = "no_llm_needed") -> Dict[str, Any]:
            reply_text = templated_answer(
                primary=primary,
                also_applies=also_applies,
                resolved_name=resolved_name,
                time_ref=time_ref,
                fetch_time=fetch_time,
                skipped_ids=skipped_ids,
                window_note=window_note,
                assumed_time=assumed_time,
                previous_change_note=prev_note,
                conditions_line=conditions_line,
                timezone_name=tz_name,
                utc_offset_seconds=utc_offset,
            )
            decision_log = list(state.get("decision_log", []))
            if decision_log:
                decision_log[-1]["answer_source"] = "template"
                decision_log[-1]["compose_reason"] = reason_text
                decision_log[-1]["reason"] = reason_text
                decision_log[-1]["model_used"] = model_used
            return {
                "kind": "advice",
                "reply": reply_text,
                "answer_source": "template",
                "reason": reason_text,
                "_compose_reason": reason_text,
                "model_used": model_used,
                "decision_log": decision_log,
            }

        if not raw_composed:
            compose_reason = state.get("_compose_reason", "no_llm_needed")
            return _fallback(reason_text=compose_reason)

        # Build complete allow-list of valid numbers
        allowed_numbers: Set[float | int] = set()

        # 1. Facts
        for val in state.get("facts", {}).values():
            if isinstance(val, (int, float)):
                allowed_numbers.add(val)
            elif isinstance(val, (list, set, tuple)):
                for item in val:
                    if isinstance(item, (int, float)):
                        allowed_numbers.add(item)

        # 2. SOP thresholds (primary + also_applies)
        if primary and primary.sop_id in self.sops_dict:
            allowed_numbers.update(extract_thresholds(self.sops_dict[primary.sop_id]))
        for r in also_applies:
            if r.sop_id in self.sops_dict:
                allowed_numbers.update(extract_thresholds(self.sops_dict[r.sop_id]))

        # 3. Window hours (e.g. 0, 3, 6, 17, 21, 22, 24) and counts
        allowed_numbers.update([0, 3, 6, 11, 16, 17, 21, 22, 24, 1, 2])
        allowed_numbers.add(len(skipped_ids))

        # Check validation
        known_ids = set(self.sops_dict.keys())
        is_valid, val_reason = validate_reply(raw_composed, allowed_numbers, known_ids)

        if not is_valid:
            # Hallucinated number or unknown SOP ID -> route to templated answer
            return _fallback(reason_text=f"validation_failed:{val_reason}")

        # Validation passed -> code adds window_note exactly once and appends citation footer
        footer = build_footer(
            primary=primary,
            also_applies=also_applies,
            resolved_name=resolved_name,
            fetch_time=fetch_time,
            skipped_ids=skipped_ids,
            conditions_line=conditions_line,
            timezone_name=tz_name,
            utc_offset_seconds=utc_offset,
        )
        if window_note:
            main_text = f"({window_note})\n\n{raw_composed.strip()}"
        else:
            main_text = raw_composed.strip()
        final_reply = main_text + footer

        decision_log = list(state.get("decision_log", []))
        if decision_log:
            decision_log[-1]["answer_source"] = "llm"
            decision_log[-1]["compose_reason"] = "llm_composed"
            decision_log[-1]["reason"] = "llm_composed"
            decision_log[-1]["model_used"] = model_used

        return {
            "kind": "advice",
            "reply": final_reply,
            "answer_source": "llm",
            "reason": "llm_composed",
            "_compose_reason": "llm_composed",
            "model_used": model_used,
            "decision_log": decision_log,
        }
