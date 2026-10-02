"""LLM Client interfaces and Google Gemini implementation.

Provides the LLMClientProtocol, typed LLMError, and GeminiClient utilizing
the official google-genai SDK with strict temperature=0, timeout, and retry handling.
Never logs secrets or user inputs.
"""

from __future__ import annotations

import os
from typing import Optional, Protocol
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env file automatically
load_dotenv()


class LLMError(Exception):
    """Raised when an LLM API request fails, times out, is blocked, or returns empty text."""
    pass


class LLMClientProtocol(Protocol):
    """Protocol defining LLM interaction boundaries."""

    def parse_intent_raw(self, system: str, user_text: str) -> str:
        """Parse user query into structured intent JSON string."""
        ...

    def compose_raw(self, system: str, payload_json: str) -> str:
        """Compose human-readable advisory from structured payload JSON string."""
        ...


class GeminiClient:
    """Production LLM client utilizing the google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        # Load API key: explicit > env > streamlit.secrets
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            try:
                import streamlit as st  # type: ignore
                if hasattr(st, "secrets"):
                    key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
            except Exception:
                pass

        if not key:
            raise LLMError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not configured in environment or secrets.")

        # Load model name: explicit > env > default
        self.model_name = model_name or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
        self.timeout_seconds = timeout_seconds

        try:
            self._client = genai.Client(api_key=key)
        except Exception as err:
            raise LLMError(f"Failed to initialize Gemini client: {err}") from err

    def parse_intent_raw(self, system: str, user_text: str) -> str:
        """Call Gemini to extract structured JSON from user text."""
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            response_mime_type="application/json",
            http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)),
        )

        # 1 retry on timeout / API failure
        last_err: Optional[Exception] = None
        for attempt in range(2):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=user_text,
                    config=config,
                )
                text = response.text
                if not text or not text.strip():
                    raise LLMError("Gemini returned empty or safety-blocked content.")
                return text.strip()
            except LLMError:
                raise
            except Exception as err:
                last_err = err

        raise LLMError(f"Gemini parse_intent failed after retry: {last_err}") from last_err

    def compose_raw(self, system: str, payload_json: str) -> str:
        """Call Gemini to compose advisory text from structured payload JSON."""
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)),
        )

        last_err: Optional[Exception] = None
        for attempt in range(2):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=payload_json,
                    config=config,
                )
                text = response.text
                if not text or not text.strip():
                    raise LLMError("Gemini returned empty or safety-blocked content during compose.")
                return text.strip()
            except LLMError:
                raise
            except Exception as err:
                last_err = err

        raise LLMError(f"Gemini compose failed after retry: {last_err}") from last_err
