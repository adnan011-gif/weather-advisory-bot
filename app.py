"""Streamlit Web UI for Weather Advisory Support Bot.

Zero-hallucination meteorological advisory assistant enforcing deterministic safety
rules, dynamic policy reloads from disk, and transparent decision audit traces.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# 1. Config: load .env from project root first
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# First Streamlit command - MUST be called before any other st command or access
st.set_page_config(
    page_title="Weather Advisory Support Bot",
    page_icon="⛅",
    layout="wide",
)

from src.config import get_config_value

# Check required API key (explicit > env > st.secrets)
api_key = get_config_value("GEMINI_API_KEY") or get_config_value("GOOGLE_API_KEY")
if not api_key:
    st.error("Missing required configuration: 'GEMINI_API_KEY' (or 'GOOGLE_API_KEY') is not set in environment or secrets.")
    st.stop()

from langgraph.checkpoint.memory import MemorySaver
from src.graph import build_graph
from src.sop_loader import load_sops
from src.weather_client import OpenMeteoClient
from src.llm_client import GeminiClient
from src.validator import FACT_UNITS, FACT_LABELS, format_local_fetch_time

# 2. Cache resources: cache client instances, NEVER SOPs
@st.cache_resource
def get_bot_clients():
    """Initialize singleton API and LLM clients and LangGraph checkpointer."""
    w_client = OpenMeteoClient()
    llm_client = GeminiClient()
    checkpointer = MemorySaver()
    return w_client, llm_client, checkpointer


# 3. Session state initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_message_time" not in st.session_state:
    st.session_state.last_message_time = 0.0

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# 4. Helper to read SOPs live from disk (NEVER cached)
def get_live_sops():
    """Read all SOP YAML files live from disk."""
    sop_path = BASE_DIR / "sops"
    try:
        return load_sops(sop_path)
    except Exception:
        return []


# 5. Sidebar layout
with st.sidebar:
    st.title("⛅ Weather Advisory Bot")
    st.caption(f"Session ID: `{st.session_state.thread_id}`")

    if st.button("🔄 New session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.session_state.last_message_time = 0.0
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown("---")
    st.subheader("💡 Example Questions")
    examples = [
        "Is it safe to cycle in Bhopal today?",
        "Can I go running in Mumbai tomorrow morning?",
        "Is it good for a picnic in Delhi this evening?",
        "Why did you say that?",
    ]
    for ex in examples:
        if st.button(ex, key=f"btn_{ex}", use_container_width=True):
            st.session_state.pending_prompt = ex
            st.rerun()

    st.markdown("---")
    st.subheader("📋 Active Policies (Live)")
    active_sops = get_live_sops()
    if active_sops:
        st.caption(f"{len(active_sops)} policies currently loaded from disk:")
        for s in active_sops:
            sev_badge = f":red[{s.severity}]" if s.severity in ("critical", "high") else f":blue[{s.severity}]"
            st.markdown(f"- **`{s.id}`** ({sev_badge}): {s.title}")
    else:
        st.warning("No policies currently loaded.")


# 6. Main chat view
st.title("Meteorological Safety Advisory Assistant")
st.write("Deterministic safety guidance based on certified Standard Operating Procedures (SOPs).")

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("metadata"):
            meta = msg["metadata"]
            with st.expander("🔍 Why this answer", expanded=False):
                st.markdown(f"**Primary Policy:** `{meta.get('primary_id', 'None')}` ({meta.get('severity', 'N/A')})")
                if meta.get("also_applies"):
                    st.markdown(f"**Also Applies:** {', '.join(meta['also_applies'])}")
                st.markdown(f"**Resolver reason:** {meta.get('resolver_reason', 'N/A')}")
                if meta.get("facts_formatted"):
                    facts_bullets = "\n".join(f"- {k}: **{v}**" for k, v in meta["facts_formatted"].items())
                    st.markdown(f"**Facts Used:**\n{facts_bullets}")
                st.markdown(f"**Location:** {meta.get('resolved_name', 'N/A')}")
                st.markdown(f"**Data Fetched:** {meta.get('fetch_time', 'N/A')}")
                st.markdown(f"**Answer source:** `{meta.get('answer_source', 'N/A')}` ({meta.get('answer_reason', 'N/A')})")
                st.markdown(f"**Model Used:** `{meta.get('model_used', 'N/A')}`")
                if meta.get("skipped_ids"):
                    st.markdown(f"**Skipped Safety Checks:** {', '.join(meta['skipped_ids'])}")


# 7. Chat input handling with rate limits and turn limits
prompt = st.chat_input("Ask about outdoor weather safety...") or st.session_state.pop("pending_prompt", None)

if prompt:
    user_query = prompt.strip()

    # Rate Limit 1: Max 500 characters
    if len(user_query) > 500:
        st.warning("Please limit your message to 500 characters.")
    # Rate Limit 2: Max 25 turns per session
    elif sum(1 for m in st.session_state.messages if m["role"] == "user") >= 25:
        st.info("Session turn limit (25 turns) reached. Please click 'New session' in the sidebar to start a fresh advisory session.")
    # Rate Limit 3: Minimum 2 seconds between messages
    elif (time.time() - st.session_state.last_message_time) < 2.0:
        st.warning("Please wait a moment before sending another message.")
    else:
        st.session_state.last_message_time = time.time()

        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Build and invoke graph
        w_client, llm_client, checkpointer = get_bot_clients()
        graph = build_graph(
            weather_client=w_client,
            llm=llm_client,
            checkpointer=checkpointer,
        )

        with st.spinner("Checking live weather and policies..."):
            try:
                state_input = {"query": user_query}
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = graph.invoke(state_input, config=config)
                reply = result.get("reply", "I was unable to produce an advisory for your request.")
            except Exception as err:
                reply = f"An unexpected error occurred while processing your request: {err}"
                result = {}

        # Extract audit metadata for the expander
        primary_data = result.get("primary") or {}
        also_data = result.get("also_applies") or []
        facts_used = primary_data.get("facts_used") or {}

        facts_formatted = {}
        for k, v in facts_used.items():
            label = FACT_LABELS.get(k, k.replace("_", " "))
            unit = FACT_UNITS.get(k, "")
            unit_str = f" {unit}" if unit and unit != "%" else unit
            facts_formatted[label] = f"{v}{unit_str}"

        # If facts_used is empty but domain facts exist, show key facts
        if not facts_formatted and result.get("facts"):
            for k in ["wind_gusts", "precipitation_sum", "uv_index_max", "apparent_temperature"]:
                if k in result["facts"] and result["facts"][k] is not None:
                    label = FACT_LABELS.get(k, k.replace("_", " "))
                    unit = FACT_UNITS.get(k, "")
                    unit_str = f" {unit}" if unit and unit != "%" else unit
                    facts_formatted[label] = f"{result['facts'][k]}{unit_str}"

        raw_fetch = result.get("raw_fetch_time", "N/A")
        formatted_fetch = format_local_fetch_time(
            raw_fetch,
            timezone_name=result.get("timezone"),
            utc_offset_seconds=result.get("utc_offset_seconds"),
        )

        also_strings = [
            f"`{r.get('sop_id')}` ({r.get('effective_severity')})"
            for r in also_data
            if r.get("match_type") != "clear"
        ]

        # Resolution reason from decision log
        decision_log = result.get("decision_log", [])
        last_turn_log = decision_log[-1] if decision_log else {}
        resolver_reason = (
            result.get("resolver_reason")
            or last_turn_log.get("resolver_reason")
            or last_turn_log.get("reason")
            or "No hazard SOP matched; clear baseline applies"
        )

        metadata = {
            "primary_id": primary_data.get("sop_id", "None"),
            "severity": primary_data.get("effective_severity", "None"),
            "also_applies": also_strings,
            "resolver_reason": resolver_reason,
            "facts_formatted": facts_formatted,
            "resolved_name": result.get("resolved_name", "N/A"),
            "fetch_time": formatted_fetch,
            "answer_source": result.get("answer_source", "template"),
            "answer_reason": result.get("reason", "N/A"),
            "model_used": result.get("model_used", "None"),
            "skipped_ids": result.get("unevaluable_ids", []),
        }

        # Save assistant message to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "metadata": metadata,
        })

        # Render assistant response immediately
        with st.chat_message("assistant"):
            st.markdown(reply)
            with st.expander("🔍 Why this answer", expanded=False):
                st.markdown(f"**Primary Policy:** `{metadata['primary_id']}` ({metadata['severity']})")
                if metadata["also_applies"]:
                    st.markdown(f"**Also Applies:** {', '.join(metadata['also_applies'])}")
                st.markdown(f"**Resolver reason:** {metadata['resolver_reason']}")
                if metadata["facts_formatted"]:
                    facts_bullets = "\n".join(f"- {k}: **{v}**" for k, v in metadata["facts_formatted"].items())
                    st.markdown(f"**Facts Used:**\n{facts_bullets}")
                st.markdown(f"**Location:** {metadata['resolved_name']}")
                st.markdown(f"**Data Fetched:** {metadata['fetch_time']}")
                st.markdown(f"**Answer source:** `{metadata['answer_source']}` ({metadata['answer_reason']})")
                st.markdown(f"**Model Used:** `{metadata['model_used']}`")
                if metadata["skipped_ids"]:
                    st.markdown(f"**Skipped Safety Checks:** {', '.join(metadata['skipped_ids'])}")
