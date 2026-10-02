# Weather-Advisory Support Bot

A LangGraph-based outdoor-activity safety advisory system powered by live Open-Meteo meteorological data and deterministic Standard Operating Procedures (SOPs).

## Core Principles
1. **Separation of Policy and Logic**: All safety rules and thresholds live in `sops/*.yaml`. Fact mapping and window definitions live in `config/facts.yaml`.
2. **Deterministic Authority**: The LLM only classifies intent and structures final wording. All hazard assessments, rule evaluations, and conflict resolution are strictly deterministic.
3. **Honest Failures**: Explicit, non-hallucinated fallback paths for geocoding misses, API downtime, passed time windows, and incomplete meteorological data.
4. **Zero Hardcoded Secrets**: Secrets loaded via environment variables (`GEMINI_API_KEY`, `GEMINI_MODEL`).
