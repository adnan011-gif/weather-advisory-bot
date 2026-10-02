"""LangGraph workflow definition for Weather Advisory Support Bot.

Constructs state graph, registers nodes, wires conditional routing edges,
and compiles with MemorySaver for session continuity across follow-up turns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import WeatherAdvisoryState
from src.facts_registry import FactsRegistry
from src.sop_loader import load_sops, derive_vocabulary
from src.weather_client import WeatherClientProtocol, OpenMeteoClient
from src.llm_client import LLMClientProtocol, GeminiClient
from src.nodes import GraphNodes


def build_graph(
    weather_client: Optional[WeatherClientProtocol] = None,
    llm: Optional[LLMClientProtocol] = None,
    sop_dir: Optional[str | Path] = None,
    facts_path: Optional[str | Path] = None,
    checkpointer: Optional[MemorySaver] = None,
):
    """Build and compile the Weather Advisory Support Bot LangGraph.

    Args:
        weather_client: Injected weather client (defaults to live OpenMeteoClient).
        llm: Injected LLM client (defaults to GeminiClient).
        sop_dir: Path to directory containing SOP YAML files.
        facts_path: Path to config/facts.yaml.
        checkpointer: Optional LangGraph checkpointer (defaults to MemorySaver()).

    Returns:
        Compiled LangGraph instance ready for .invoke({"query": ...}, config={"configurable": {"thread_id": ...}}).
    """
    base_dir = Path(__file__).resolve().parent.parent
    sop_path = Path(sop_dir) if sop_dir else (base_dir / "sops")
    config_file = Path(facts_path) if facts_path else (base_dir / "config" / "facts.yaml")

    facts_reg = FactsRegistry(config_path=config_file)
    sops = load_sops(sop_path, facts_reg)
    vocabulary = derive_vocabulary(sops)

    w_client = weather_client or OpenMeteoClient(facts_registry=facts_reg)
    llm_client = llm or GeminiClient()
    memory = checkpointer if checkpointer is not None else MemorySaver()

    nodes = GraphNodes(
        weather_client=w_client,
        llm=llm_client,
        sops=sops,
        facts_registry=facts_reg,
        vocabulary=vocabulary,
    )

    workflow = StateGraph(WeatherAdvisoryState)

    # Add Nodes
    workflow.add_node("parse_intent", nodes.parse_intent_node)
    workflow.add_node("no_guidance", nodes.no_guidance_node)
    workflow.add_node("explain_decision", nodes.explain_decision_node)
    workflow.add_node("resolve_context", nodes.resolve_context_node)
    workflow.add_node("geocode", nodes.geocode_node)
    workflow.add_node("fetch_weather", nodes.fetch_weather_node)
    workflow.add_node("compute_facts", nodes.compute_facts_node)
    workflow.add_node("match_sops", nodes.match_sops_node)
    workflow.add_node("resolve_conflicts", nodes.resolve_conflicts_node)
    workflow.add_node("compose", nodes.compose_node)
    workflow.add_node("validate_answer", nodes.validate_answer_node)

    # 1. Start -> parse_intent
    workflow.add_edge(START, "parse_intent")

    # 2. Conditional edge after parse_intent
    def _route_after_parse(state: WeatherAdvisoryState) -> str:
        if state.get("kind") == "parse_failed":
            return END
        intent = state.get("_parsed_intent")
        if intent == "out_of_scope":
            return "no_guidance"
        elif intent == "explain":
            return "explain_decision"
        elif intent == "advice":
            return "resolve_context"
        return END

    workflow.add_conditional_edges(
        "parse_intent",
        _route_after_parse,
        {
            END: END,
            "no_guidance": "no_guidance",
            "explain_decision": "explain_decision",
            "resolve_context": "resolve_context",
        },
    )

    # 3. no_guidance & explain_decision -> END
    workflow.add_edge("no_guidance", END)
    workflow.add_edge("explain_decision", END)

    # 4. Conditional edge after resolve_context
    def _route_after_resolve(state: WeatherAdvisoryState) -> str:
        if state.get("kind") in ("ask_location", "ask_activity"):
            return END
        return "geocode"

    workflow.add_conditional_edges(
        "resolve_context",
        _route_after_resolve,
        {
            END: END,
            "geocode": "geocode",
        },
    )

    # 5. Conditional edge after geocode
    def _route_after_geocode(state: WeatherAdvisoryState) -> str:
        if state.get("kind") == "format_error":
            return END
        return "fetch_weather"

    workflow.add_conditional_edges(
        "geocode",
        _route_after_geocode,
        {
            END: END,
            "fetch_weather": "fetch_weather",
        },
    )

    # 6. Conditional edge after fetch_weather
    def _route_after_fetch(state: WeatherAdvisoryState) -> str:
        if state.get("kind") == "format_error":
            return END
        return "compute_facts"

    workflow.add_conditional_edges(
        "fetch_weather",
        _route_after_fetch,
        {
            END: END,
            "compute_facts": "compute_facts",
        },
    )

    # 7. Conditional edge after compute_facts
    def _route_after_compute(state: WeatherAdvisoryState) -> str:
        if state.get("kind") == "window_passed":
            return END
        return "match_sops"

    workflow.add_conditional_edges(
        "compute_facts",
        _route_after_compute,
        {
            END: END,
            "match_sops": "match_sops",
        },
    )

    # 8. Conditional edge after match_sops
    def _route_after_match(state: WeatherAdvisoryState) -> str:
        if state.get("kind") in ("data_incomplete", "no_guidance"):
            return END
        return "resolve_conflicts"

    workflow.add_conditional_edges(
        "match_sops",
        _route_after_match,
        {
            END: END,
            "resolve_conflicts": "resolve_conflicts",
        },
    )

    # 9. resolve_conflicts -> compose -> validate_answer -> END
    workflow.add_edge("resolve_conflicts", "compose")
    workflow.add_edge("compose", "validate_answer")
    workflow.add_edge("validate_answer", END)

    return workflow.compile(checkpointer=memory)
