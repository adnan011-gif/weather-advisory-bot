"""Live smoke test script for Open-Meteo geocoding and forecast retrieval.

Geocodes 'Bhopal', fetches live weather data, computes facts for window 'today',
and prints all resolved metrics with their configured units.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so scripts can run standalone
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.facts_registry import FactsRegistry
from src.weather_client import OpenMeteoClient


def main() -> None:
    city = "Bhopal"
    print(f"=== Live Weather Smoke Test: {city} ===")

    registry = FactsRegistry()
    client = OpenMeteoClient(facts_registry=registry)

    # 1. Geocoding
    print(f"[1] Geocoding '{city}'...")
    geocode_res = client.geocode(city)
    print(f"    Resolved Name: {geocode_res.resolved_name}")
    print(f"    Coordinates:   {geocode_res.lat}, {geocode_res.lon}")
    print(f"    Timezone:      {geocode_res.timezone}")

    # 2. Fetch Weather
    print("\n[2] Fetching live weather forecast from Open-Meteo...")
    weather_raw = client.fetch_weather(geocode_res.lat, geocode_res.lon)
    print(f"    Fetch Time (UTC): {weather_raw.fetch_time}")

    # 3. Compute Facts for 'today'
    print("\n[3] Computing facts for window 'today'...")
    computed = client.compute_facts(weather_raw, window_name="today")
    print(f"    Local Timezone:       {computed.timezone} (UTC+{computed.utc_offset_seconds/3600:g}h)")
    print(f"    Partly Passed Window: {computed.partly_passed}")
    if computed.window_note:
        print(f"    Window Note:          {computed.window_note}")

    print("\n[4] Evaluated Facts Registry Values:")
    print("-" * 75)
    print(f"{'Fact Name':<28} | {'Source':<8} | {'Agg':<8} | {'Value':<14} | {'Unit':<8}")
    print("-" * 75)

    for fact_def in registry.config.facts:
        val = computed.facts.get(fact_def.name)
        val_str = str(val) if val is not None else "None"
        unit_str = fact_def.unit if fact_def.unit else "-"
        print(f"{fact_def.name:<28} | {fact_def.source:<8} | {fact_def.agg:<8} | {val_str:<14} | {unit_str:<8}")
    print("-" * 75)
    print("Smoke test successfully completed.")


if __name__ == "__main__":
    main()
