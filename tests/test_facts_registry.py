"""Unit tests for facts registry, window computations, and weather client.

All tests operate with hand-made payloads and mock transports (no external network calls).
"""

from __future__ import annotations

import datetime
from pathlib import Path
import pytest
import httpx
import yaml

from src.facts_registry import FactsRegistry, FactsConfig
from src.weather_client import (
    OpenMeteoClient,
    LocationError,
    WeatherAPIError,
    WindowPassedError,
    sanitize_location,
    RawWeatherData,
)


@pytest.fixture
def registry() -> FactsRegistry:
    """Fixture providing initialized FactsRegistry from config/facts.yaml."""
    return FactsRegistry()


@pytest.fixture
def sample_payload() -> dict:
    """Mock Open-Meteo payload for testing aggregation."""
    # 24 hours of data for 2026-10-02 (00:00 to 23:00) in UTC+0
    times = [f"2026-10-02T{h:02d}:00" for h in range(24)]
    temps = [15.0 + float(h) for h in range(24)]  # 15.0 to 38.0
    wind_speeds = [10.0 + float(h) for h in range(24)]
    wind_gusts = [20.0 + float(h) for h in range(24)]
    precip = [0.0 if h % 2 == 0 else 1.5 for h in range(24)]
    precip_prob = [float(h * 3) for h in range(24)]
    uv = [0.0] * 11 + [5.0, 6.0, 7.0, 6.0, 4.0, 1.0] + [0.0] * 7  # UV peak around noon
    humidity = [60.0 for _ in range(24)]
    pressure = [1013.25 for _ in range(24)]

    return {
        "latitude": 47.6062,
        "longitude": -122.3321,
        "utc_offset_seconds": 0,
        "timezone": "UTC",
        "current": {
            "weather_code": 1,
            "temperature_2m": 18.5,
        },
        "hourly": {
            "time": times,
            "temperature_2m": temps,
            "apparent_temperature": [t + 1.0 for t in temps],
            "wind_speed_10m": wind_speeds,
            "wind_gusts_10m": wind_gusts,
            "precipitation": precip,
            "precipitation_probability": precip_prob,
            "uv_index": uv,
            "weather_code": [0, 1, 2, 3] * 6,
            "relative_humidity_2m": humidity,
            "pressure_msl": pressure,
        },
    }


def test_normal_aggregation_for_each_agg_type(registry: FactsRegistry, sample_payload: dict):
    """Test standard aggregation operations (current, max, min, sum, mean, set)."""
    client = OpenMeteoClient(facts_registry=registry)
    # Simulate local time at 12:00 on 2026-10-02
    now_local = datetime.datetime(2026, 10, 2, 12, 0, tzinfo=datetime.timezone.utc)

    # Test "now" window (12:00 to 15:00: hours 12, 13, 14, 15)
    result = client.compute_facts(sample_payload, window_name="now", now_local=now_local)

    # set agg: weather_code from hourly section for hours 12..15
    assert result.facts["weather_code"] == [0, 1, 2, 3]

    # max agg: wind_speed max for hours 12..15 -> max(10+12, 10+13, 10+14, 10+15) = 25.0
    assert result.facts["wind_speed"] == 25.0

    # min agg: check helper directly or verify computed
    assert client._aggregate([10.0, 5.5, 20.0], "min") == 5.5

    # sum agg: precipitation sum for hours 12, 13, 14, 15
    # indices: 12 (even: 0.0), 13 (odd: 1.5), 14 (even: 0.0), 15 (odd: 1.5) -> sum = 3.0
    assert result.facts["precipitation_sum"] == 3.0

    # max agg: temperature max for hours 12, 13, 14, 15
    # temps: 27, 28, 29, 30 -> max = 30.0
    assert result.facts["temperature"] == 30.0

    # mean agg: humidity
    assert result.facts["humidity"] == 60.0

    # min agg: pressure_msl
    assert result.facts["pressure_msl"] == 1013.2


def test_window_hourly_inclusivity(registry: FactsRegistry, sample_payload: dict):
    """Test start-inclusive and end-exclusive hourly window semantics:
    - at 14:15 the 14:00 hour is included for 'today'
    - at 19:15 the 19:00 hour is included for 'this_evening'
    - the 21:00 hour is excluded for 'this_evening'
    """
    client = OpenMeteoClient(facts_registry=registry)

    # 1. At 14:15, test that 14:00 is included in 'today'
    # In sample_payload, hour 14 has temperature 15 + 14 = 29.0
    # Let's set temperature for hours 0..13 to None or low, hour 14 to 99.0
    payload_today = dict(sample_payload)
    hourly_copy = dict(sample_payload["hourly"])
    hourly_copy["temperature_2m"] = [99.0 if h == 14 else 10.0 for h in range(24)]
    payload_today["hourly"] = hourly_copy

    now_1415 = datetime.datetime(2026, 10, 2, 14, 15, tzinfo=datetime.timezone.utc)
    res_today = client.compute_facts(payload_today, window_name="today", now_local=now_1415)
    # If 14:00 was excluded, max would be 10.0. Since 14:00 is included, max is 99.0!
    assert res_today.facts["temperature"] == 99.0

    # 2. At 19:15, test that 19:00 is included in 'this_evening' and 21:00 is excluded
    # For this_evening (17:00 - 21:00):
    # Set hour 19 to 50.0, hour 20 to 10.0, hour 21 to 100.0
    hourly_eve = dict(sample_payload["hourly"])
    hourly_eve["temperature_2m"] = [
        50.0 if h == 19 else (100.0 if h == 21 else 10.0) for h in range(24)
    ]
    payload_today["hourly"] = hourly_eve

    now_1915 = datetime.datetime(2026, 10, 2, 19, 15, tzinfo=datetime.timezone.utc)
    res_eve = client.compute_facts(payload_today, window_name="this_evening", now_local=now_1915)
    # Hour 19:00 is included (50.0). Hour 21:00 is EXCLUDED (not 100.0). Max should be 50.0.
    assert res_eve.facts["temperature"] == 50.0


def test_null_hourly_values_and_all_null(registry: FactsRegistry):
    """Test that null values are skipped and all-null series aggregate to None."""
    client = OpenMeteoClient(facts_registry=registry)
    now_local = datetime.datetime(2026, 10, 2, 12, 0, tzinfo=datetime.timezone.utc)

    times = [f"2026-10-02T{h:02d}:00" for h in range(24)]
    # Partial nulls in temperature
    temps = [None if h in (12, 13) else 20.0 for h in range(24)]
    # All nulls in wind_speed
    wind_speeds = [None for _ in range(24)]

    payload = {
        "utc_offset_seconds": 0,
        "timezone": "UTC",
        "current": {"weather_code": 0},
        "hourly": {
            "time": times,
            "temperature_2m": temps,
            "wind_speed_10m": wind_speeds,
        },
    }

    result = client.compute_facts(payload, window_name="now", now_local=now_local)
    # Temperature for hours 12..15 has [None, None, 20.0, 20.0] -> mean = 20.0
    assert result.facts["temperature"] == 20.0
    # Wind speed is all None -> None
    assert result.facts["wind_speed"] is None


def test_empty_arrays_and_missing_keys(registry: FactsRegistry):
    """Test that empty hourly structures or missing variables safely evaluate to None."""
    client = OpenMeteoClient(facts_registry=registry)
    now_local = datetime.datetime(2026, 10, 2, 12, 0, tzinfo=datetime.timezone.utc)

    payload = {
        "utc_offset_seconds": 0,
        "timezone": "UTC",
        "hourly": {},  # empty
    }

    result = client.compute_facts(payload, window_name="now", now_local=now_local)
    assert result.facts["temperature"] is None
    assert result.facts["wind_speed"] is None
    assert result.facts["weather_code"] is None


def test_window_fully_passed(registry: FactsRegistry, sample_payload: dict):
    """Test that a fully elapsed window raises WindowPassedError."""
    client = OpenMeteoClient(facts_registry=registry)
    # this_evening is 17:00-21:00. Now is 21:30 local time.
    now_local = datetime.datetime(2026, 10, 2, 21, 30, tzinfo=datetime.timezone.utc)

    with pytest.raises(WindowPassedError) as exc_info:
        client.compute_facts(sample_payload, window_name="this_evening", now_local=now_local)
    assert "has already ended" in str(exc_info.value)


def test_window_partly_passed(registry: FactsRegistry, sample_payload: dict):
    """Test that a partly elapsed window evaluates the remainder and sets partly_passed flag."""
    client = OpenMeteoClient(facts_registry=registry)
    # this_evening is 17:00-21:00. Now is 19:00 local time.
    now_local = datetime.datetime(2026, 10, 2, 19, 0, tzinfo=datetime.timezone.utc)

    result = client.compute_facts(sample_payload, window_name="this_evening", now_local=now_local)
    assert result.partly_passed is True
    assert result.window_note is not None
    assert "19:00 to 21:00" in result.window_note

    # Hours evaluated: 19, 20, 21. Temps: 15+19=34, 15+20=35, 15+21=36 -> mean = 35.0
    assert result.facts["temperature"] == 35.0


def test_fact_added_only_in_facts_yaml_appears_in_request_variables(tmp_path: Path):
    """Verify that adding a fact in facts.yaml immediately updates query variables without code changes."""
    yaml_content = {
        "windows": {
            "now": {"type": "relative_hours", "start_offset_hours": 0, "end_offset_hours": 3}
        },
        "facts": [
            {
                "name": "soil_moisture",
                "source": "hourly",
                "open_meteo_var": "soil_moisture_0_to_1cm",
                "agg": "mean",
                "unit": "m³/m³",
            }
        ],
    }
    custom_yaml = tmp_path / "custom_facts.yaml"
    with open(custom_yaml, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f)

    custom_registry = FactsRegistry(config_path=custom_yaml)
    vars_dict = custom_registry.get_query_variables()

    assert "soil_moisture_0_to_1cm" in vars_dict["hourly"]


def test_location_sanitization():
    """Verify location sanitization handles whitespace, truncation, and empty inputs."""
    # Normal stripping
    assert sanitize_location("  Paris, France   ") == "Paris, France"

    # Truncation at 100 characters
    long_loc = "A" * 120
    sanitized = sanitize_location(long_loc)
    assert len(sanitized) == 100
    assert sanitized == "A" * 100

    # Empty inputs raise LocationError
    with pytest.raises(LocationError):
        sanitize_location("")

    with pytest.raises(LocationError):
        sanitize_location("   \n\t   ")


def test_geocode_error_handling_fake_transport(registry: FactsRegistry):
    """Test geocode failure branches (empty results, timeout, HTTP 500, malformed JSON) with mock transport."""
    # 1. Empty results
    def handler_empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    client = OpenMeteoClient(facts_registry=registry, http_client=httpx.Client(transport=httpx.MockTransport(handler_empty)))
    with pytest.raises(LocationError) as exc_info:
        client.geocode("AtlantisNonExistentCity")
    assert "Could not find coordinates" in str(exc_info.value)

    # 2. Timeout
    def handler_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Connection timed out")

    client_timeout = OpenMeteoClient(facts_registry=registry, http_client=httpx.Client(transport=httpx.MockTransport(handler_timeout)))
    with pytest.raises(LocationError) as exc_info:
        client_timeout.geocode("London")
    assert "Geocoding request failed" in str(exc_info.value)

    # 3. HTTP 500
    def handler_500(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    client_500 = OpenMeteoClient(facts_registry=registry, http_client=httpx.Client(transport=httpx.MockTransport(handler_500)))
    with pytest.raises(LocationError) as exc_info:
        client_500.geocode("London")
    assert "returned HTTP 500" in str(exc_info.value)

    # 4. Bad JSON
    def handler_bad_json(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="NOT_VALID_JSON{")

    client_bad_json = OpenMeteoClient(facts_registry=registry, http_client=httpx.Client(transport=httpx.MockTransport(handler_bad_json)))
    with pytest.raises(LocationError):
        client_bad_json.geocode("London")


def test_weather_error_handling_fake_transport(registry: FactsRegistry):
    """Test weather API failure branches (timeout, HTTP 500) with mock transport."""
    # 1. Timeout
    def handler_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Weather API timed out")

    client_timeout = OpenMeteoClient(facts_registry=registry, http_client=httpx.Client(transport=httpx.MockTransport(handler_timeout)))
    with pytest.raises(WeatherAPIError) as exc_info:
        client_timeout.fetch_weather(47.6, -122.3)
    assert "Weather API request failed" in str(exc_info.value)

    # 2. HTTP 500
    def handler_500(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Upstream Forecast Down")

    client_500 = OpenMeteoClient(facts_registry=registry, http_client=httpx.Client(transport=httpx.MockTransport(handler_500)))
    with pytest.raises(WeatherAPIError) as exc_info:
        client_500.fetch_weather(47.6, -122.3)
    assert "returned HTTP 500" in str(exc_info.value)
