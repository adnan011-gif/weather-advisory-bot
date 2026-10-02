"""Configuration resolution utilities.

Resolves configuration values across:
1. Explicit function arguments
2. Environment variables (os.environ, loaded via dotenv)
3. Streamlit secrets (st.secrets), accessed lazily and safely
"""

from __future__ import annotations

import os
from typing import Optional


def get_secret_lazily(key: str) -> Optional[str]:
    """Safely and lazily fetch a secret from st.secrets.

    Never raises FileNotFoundError or accesses st.secrets at import time.
    Returns None if Streamlit is not installed, no secrets file exists, or key is missing.
    """
    try:
        import streamlit as st  # type: ignore

        if not hasattr(st, "secrets"):
            return None
        val = st.secrets.get(key)
        return str(val) if val is not None else None
    except Exception:
        return None


def get_config_value(
    key: str,
    explicit_val: Optional[str] = None,
    default: Optional[str] = None,
) -> Optional[str]:
    """Resolve a configuration parameter in strict priority order:
    1. Explicit value (if provided and not None)
    2. os.environ
    3. st.secrets (lazily and safely)
    4. default value

    Never logs or exposes secret values.
    """
    if explicit_val is not None and str(explicit_val).strip() != "":
        return explicit_val

    env_val = os.getenv(key)
    if env_val is not None and env_val.strip() != "":
        return env_val

    secret_val = get_secret_lazily(key)
    if secret_val is not None and secret_val.strip() != "":
        return secret_val

    return default
