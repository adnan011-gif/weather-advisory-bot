"""Tests for config helper:
1. Returns env values when set.
2. Explicit values take precedence over env.
3. Returns None without raising when neither env nor secrets exist.
4. Does not import streamlit at module import time.
"""

from __future__ import annotations

import os
import subprocess
import sys

from src.config import get_config_value, get_secret_lazily


def test_config_helper_returns_env_values_when_set():
    """Verify that get_config_value retrieves values from os.environ."""
    test_key = "TEST_WEATHER_BOT_ENV_VAR"
    os.environ[test_key] = "test_environment_value"
    try:
        val = get_config_value(test_key)
        assert val == "test_environment_value"
    finally:
        os.environ.pop(test_key, None)


def test_config_helper_explicit_takes_precedence():
    """Verify that explicit arguments beat environment variables."""
    test_key = "TEST_WEATHER_BOT_ENV_VAR_PRECEDENCE"
    os.environ[test_key] = "env_value"
    try:
        val = get_config_value(test_key, explicit_val="explicit_override")
        assert val == "explicit_override"
    finally:
        os.environ.pop(test_key, None)


def test_config_helper_returns_none_when_neither_env_nor_secrets_exist():
    """Verify that nonexistent keys safely return None without raising."""
    nonexistent = "NONEXISTENT_KEY_DEFINITELY_NOT_SET_12345"
    os.environ.pop(nonexistent, None)

    # Must return None without raising FileNotFoundError or Streamlit exceptions
    val = get_config_value(nonexistent)
    assert val is None

    # Default fallback
    val_with_default = get_config_value(nonexistent, default="fallback_default")
    assert val_with_default == "fallback_default"

    # Direct lazy secret getter returns None
    secret_val = get_secret_lazily(nonexistent)
    assert secret_val is None


def test_config_does_not_import_streamlit_at_module_import_time():
    """Verify in a clean process that importing src.config does not import streamlit."""
    test_script = """
import sys
import src.config
if 'streamlit' in sys.modules:
    sys.exit(1)
sys.exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", test_script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "src.config must not import streamlit at module import time."
