"""Interactive CLI terminal for testing Weather Advisory Support Bot with session memory."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Load .env variables (e.g. GEMINI_API_KEY, GEMINI_MODEL)
load_dotenv(BASE_DIR / ".env")

from src.graph import build_graph


def run_cli(thread_id: str = "cli-session-1") -> None:
    print("=" * 65)
    print("  Weather-Advisory Support Bot - Interactive Terminal Session")
    print(f"  Session Thread ID: {thread_id}")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 65 + "\n")

    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\nUser > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting session.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        state_input = {"query": user_input}
        try:
            result = graph.invoke(state_input, config=config)
            if os.getenv("DEBUG_REASONS") == "1":
                reason_str = result.get("error_reason") or "none"
                model_str = result.get("model_used") or "none"
                print(f"[debug] reason: {reason_str}, model: {model_str}")
            reply = result.get("reply", "No response generated.")
            print(f"\nBot > {reply}")
        except Exception as err:
            print(f"\n[System Error] {err}")


if __name__ == "__main__":
    session_id = sys.argv[1] if len(sys.argv) > 1 else "cli-session-1"
    run_cli(thread_id=session_id)
