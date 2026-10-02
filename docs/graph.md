# LangGraph Workflow Architecture

This document visualizes the compiled state machine for the Weather Advisory Support Bot.

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
	__start__([__start__]):::first
	parse_intent(parse_intent - LLM)
	parse_failed(parse_failed - Deterministic)
	llm_unavailable(llm_unavailable - Deterministic)
	request_blocked(request_blocked - Deterministic)
	no_guidance(no_guidance - Deterministic)
	explain_decision(explain_decision - Deterministic)
	resolve_context(resolve_context - Deterministic)
	geocode(geocode - Deterministic)
	fetch_weather(fetch_weather - Deterministic)
	compute_facts(compute_facts - Deterministic)
	match_sops(match_sops - Deterministic)
	resolve_conflicts(resolve_conflicts - Deterministic)
	compose(compose - LLM)
	validate_answer(validate_answer - Deterministic)
	__end__([__end__]):::last

	__start__ --> parse_intent;
	parse_failed --> __end__;
	llm_unavailable --> __end__;
	request_blocked --> __end__;
	explain_decision --> __end__;
	no_guidance --> __end__;
	compose --> validate_answer;
	resolve_conflicts --> compose;
	validate_answer --> __end__;

	parse_intent -.-> parse_failed;
	parse_intent -.-> llm_unavailable;
	parse_intent -.-> request_blocked;
	parse_intent -.-> no_guidance;
	parse_intent -.-> explain_decision;
	parse_intent -.-> resolve_context;

	resolve_context -.-> __end__;
	resolve_context -.-> geocode;

	geocode -.-> __end__;
	geocode -.-> fetch_weather;

	fetch_weather -.-> __end__;
	fetch_weather -.-> compute_facts;

	compute_facts -.-> __end__;
	compute_facts -.-> match_sops;

	match_sops -.-> __end__;
	match_sops -.-> resolve_conflicts;

	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

## Routing & Node Responsibilities

1. **`parse_intent` (LLM)**: Extracts location, activity tags (restricted to derived vocabulary), and time reference from untrusted user query. Handles model fallback and daily quota cooldown.
2. **`parse_failed` (Deterministic)**: Handles unparseable JSON or schema violations with polite rephrase prompt.
3. **`llm_unavailable` (Deterministic)**: Handles LLM API errors, rate limits, timeouts, or quota exhaustion with honest service-limit message.
4. **`request_blocked` (Deterministic)**: Handles safety-blocked requests honestly without model switching.
5. **`no_guidance` (Deterministic)**: Emits a polite out-of-scope response.
6. **`explain_decision` (Deterministic)**: Answers "why did you say that?" from session audit log (`decision_log`).
7. **`resolve_context` (Deterministic)**: Reuses prior session location/activity if omitted in follow-ups; defaults missing time to `today`.
8. **`geocode` (Deterministic)**: Resolves coordinates via Open-Meteo Geocoding API; routes to `format_error` on failure.
9. **`fetch_weather` (Deterministic)**: Fetches fresh live metrics; routes to `format_error` on network/API failure.
10. **`compute_facts` (Deterministic)**: Aggregates facts for local time window; routes to `window_passed` if window elapsed.
11. **`match_sops` (Deterministic)**: Evaluates SOP conditions with 3-valued logic; routes to `data_incomplete` or `no_guidance` if 0 match.
12. **`resolve_conflicts` (Deterministic)**: Resolves primary advisory via strict precedence (override $\to$ severity $\to$ priority $\to$ id).
13. **`compose` (LLM)**: Drafts human response from structured advice only (receives zero raw user input). Falls back to `templated_answer` on error.
14. **`validate_answer` (Deterministic)**: Verifies numbers and SOP IDs against allow-list; appends code-generated citation footer. Falls back to `templated_answer` if validation fails.
