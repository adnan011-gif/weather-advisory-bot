"""LLM Client interfaces and Google Gemini implementation.

Provides the LLMClientProtocol, typed LLMError, and GeminiClient utilizing
the official google-genai SDK with strict temperature=0, timeout, and retry handling.
Never logs secrets or user inputs.
"""

from __future__ import annotations

import os
import re
import time
from typing import List, Optional, Protocol
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env file automatically
load_dotenv()


class LLMError(Exception):
    """Raised when an LLM API request fails, times out, is blocked, or returns empty text."""

    def __init__(self, message: str, reason: str = "api_error:unknown") -> None:
        super().__init__(message)
        self.reason = reason


def classify_llm_exception(err: Exception) -> str:
    """Extract a short machine reason from an LLM exception without user text or secrets."""
    err_str = str(err).lower()

    # 1. 404 Not Found
    code = getattr(err, "code", None)
    if code == 404 or "404" in err_str or "not_found" in err_str:
        return "api_error:404"

    # 2. Rate limiting (429)
    if code == 429 or "429" in err_str or "resource_exhausted" in err_str or "rate limit" in err_str:
        return "rate_limited"

    # 3. Timeout
    if "timeout" in err_str or isinstance(err, TimeoutError):
        return "api_error:timeout"

    # 4. Safety filters
    if "safety" in err_str or "blocked" in err_str:
        return "safety_blocked"

    # 5. Status code from exception object
    if code is not None:
        return f"api_error:{code}"

    # 6. Regex match for HTTP 4xx/5xx status code in error message
    status_match = re.search(r"\b([45]\d{2})\b", str(err))
    if status_match:
        return f"api_error:{status_match.group(1)}"

    return "api_error:unknown"


def parse_fallback_models(primary: str, fallback_str: Optional[str]) -> List[str]:
    """Parse, trim whitespace, filter empty entries, and deduplicate fallback models."""
    if not fallback_str:
        return []
    parts = [p.strip() for p in fallback_str.split(",")]
    result: List[str] = []
    seen: set[str] = {primary}
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            result.append(p)
    return result


class DailyQuotaCooldownTracker:
    """Tracks models on in-memory cooldown when daily quota is exhausted."""

    def __init__(self) -> None:
        self._exhausted_models: set[str] = set()

    def is_on_cooldown(self, model: str) -> bool:
        """Return True if model is on daily-quota cooldown."""
        return model in self._exhausted_models

    def mark_cooldown(self, model: str) -> None:
        """Put a model on daily-quota cooldown."""
        self._exhausted_models.add(model)

    def clear(self) -> None:
        """Clear all cooldowns."""
        self._exhausted_models.clear()

    @staticmethod
    def is_daily_quota_error(err: Exception) -> bool:
        """Check if an exception indicates that daily quota is exhausted ('PerDay' in message)."""
        msg = str(err)
        return "PerDay" in msg or "perday" in msg.lower()


# Shared singleton cooldown tracker across requests
GLOBAL_COOLDOWN_TRACKER = DailyQuotaCooldownTracker()


class LLMClientProtocol(Protocol):
    """Protocol defining LLM interaction boundaries."""

    @property
    def model_used(self) -> Optional[str]:
        ...

    def parse_intent_raw(self, system: str, user_text: str) -> str:
        """Parse user query into structured intent JSON string."""
        ...

    def compose_raw(self, system: str, payload_json: str) -> str:
        """Compose human-readable advisory from structured payload JSON string."""
        ...


class GeminiClient:
    """Production LLM client utilizing google-genai SDK with model fallback and quota cooldown."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        fallback_models: Optional[List[str] | str] = None,
        timeout_seconds: float = 20.0,
        cooldown_tracker: Optional[DailyQuotaCooldownTracker] = None,
        backoffs: Optional[List[float]] = None,
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
            raise LLMError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not configured in environment or secrets.", reason="api_error:auth")

        # Load primary model name: explicit > env > streamlit.secrets > default
        primary = model_name or os.getenv("GEMINI_MODEL")
        if not primary:
            try:
                import streamlit as st  # type: ignore
                if hasattr(st, "secrets"):
                    primary = st.secrets.get("GEMINI_MODEL")
            except Exception:
                pass
        self.primary_model = primary or "gemini-2.5-flash"
        self.model_name = self.primary_model

        # Load fallback models: explicit > env > streamlit.secrets > empty
        fb_raw = fallback_models
        if fb_raw is None:
            fb_raw = os.getenv("GEMINI_FALLBACK_MODELS")
        if fb_raw is None:
            try:
                import streamlit as st  # type: ignore
                if hasattr(st, "secrets"):
                    fb_raw = st.secrets.get("GEMINI_FALLBACK_MODELS")
            except Exception:
                pass

        if isinstance(fb_raw, list):
            self.fallback_models = [m for m in fb_raw if m != self.primary_model]
        else:
            self.fallback_models = parse_fallback_models(self.primary_model, fb_raw)

        self.model_chain = [self.primary_model] + self.fallback_models
        self.timeout_seconds = timeout_seconds
        self.cooldown_tracker = cooldown_tracker or GLOBAL_COOLDOWN_TRACKER
        self.backoffs = backoffs if backoffs is not None else [1.0, 2.0]
        self._model_used: Optional[str] = None

        try:
            self._client = genai.Client(api_key=key)
        except Exception as err:
            reason = classify_llm_exception(err)
            raise LLMError(f"Failed to initialize Gemini client: {reason}", reason=reason) from err

    @property
    def model_used(self) -> Optional[str]:
        return self._model_used

    def _call_model_chain(self, contents: str, config: types.GenerateContentConfig) -> str:
        """Call models in chain with cooldown checks, retry backoffs, and error handling."""
        last_err: Optional[Exception] = None
        last_reason: str = "api_error:unknown"

        for model in self.model_chain:
            # 1. Skip models on daily cooldown with 0 attempts
            if self.cooldown_tracker.is_on_cooldown(model):
                continue

            # Up to 3 attempts
            for attempt in range(3):
                if attempt > 0:
                    delay = self.backoffs[attempt - 1] if attempt - 1 < len(self.backoffs) else 2.0
                    if delay > 0:
                        time.sleep(delay)

                try:
                    response = self._client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    text = response.text
                    if not text or not text.strip():
                        is_safety = False
                        if hasattr(response, "candidates") and response.candidates:
                            cand = response.candidates[0]
                            finish_reason = str(getattr(cand, "finish_reason", "")).upper()
                            if "SAFETY" in finish_reason or "BLOCK" in finish_reason:
                                is_safety = True
                        if is_safety:
                            raise LLMError("Gemini response was blocked by safety filters.", reason="safety_blocked")
                        raise LLMError("Gemini returned empty content.", reason="empty_response")

                    self._model_used = model
                    return text.strip()

                except LLMError as err:
                    if err.reason == "safety_blocked":
                        # Safety block -> NEVER switch models or retry
                        self._model_used = None
                        raise
                    last_err = err
                    last_reason = err.reason
                    # Empty response is not 429/503/timeout -> move to next model
                    break

                except Exception as err:
                    last_err = err
                    last_reason = classify_llm_exception(err)

                    # 404: model unavailable -> skip immediately with no retries
                    if last_reason == "api_error:404" or getattr(err, "code", None) == 404 or "404" in str(err):
                        break

                    # 429 Daily Quota:
                    if (last_reason == "rate_limited" or getattr(err, "code", None) == 429) and self.cooldown_tracker.is_daily_quota_error(err):
                        self.cooldown_tracker.mark_cooldown(model)
                        break

                    # Only retry on 429 (per-minute), 503, or timeout
                    is_retryable = (
                        last_reason == "rate_limited"
                        or last_reason == "api_error:503"
                        or last_reason == "api_error:timeout"
                        or getattr(err, "code", None) in (429, 503)
                    )
                    if not is_retryable:
                        break

        self._model_used = None
        raise LLMError(f"All Gemini models in chain failed: {last_reason}", reason=last_reason) from last_err

    def parse_intent_raw(self, system: str, user_text: str) -> str:
        """Call Gemini to extract structured JSON from user text."""
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            response_mime_type="application/json",
            http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)),
        )
        return self._call_model_chain(contents=user_text, config=config)

    def compose_raw(self, system: str, payload_json: str) -> str:
        """Call Gemini to compose advisory text from structured payload JSON."""
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)),
        )
        return self._call_model_chain(contents=payload_json, config=config)
