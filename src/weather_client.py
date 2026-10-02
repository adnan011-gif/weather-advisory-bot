"""Weather client module for Open-Meteo geocoding and forecast APIs.

Implements synchronous HTTP requests via httpx with strict timeout, one retry on timeout,
URL encoding, facts-driven parameter construction, and local-timezone window aggregation.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Protocol
import httpx
from pydantic import BaseModel, Field

from src.facts_registry import FactsRegistry


class LocationError(Exception):
    """Raised when geocoding fails or location is invalid/empty."""
    pass


class WeatherAPIError(Exception):
    """Raised when the weather API call fails (HTTP error, timeout, malformed payload)."""
    pass


class WindowPassedError(Exception):
    """Raised when the requested time window has fully elapsed in the local timezone."""
    pass


class GeocodeResult(BaseModel):
    """Structured result of a successful geocoding lookup."""
    lat: float
    lon: float
    resolved_name: str
    timezone: str = "UTC"
    country: Optional[str] = None
    admin1: Optional[str] = None


class RawWeatherData(BaseModel):
    """Wrapper holding raw Open-Meteo response and retrieval metadata."""
    payload: Dict[str, Any]
    fetch_time: str = Field(description="ISO-8601 UTC timestamp of when data was fetched")


class ComputedFactsResult(BaseModel):
    """Computed domain facts and window metadata."""
    facts: Dict[str, Optional[Any]]
    window_name: str
    partly_passed: bool = False
    window_note: Optional[str] = None
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    fetch_time: str
    timezone: str
    utc_offset_seconds: int


class WeatherClientProtocol(Protocol):
    """Protocol defining the interface for Open-Meteo interactions."""

    def geocode(self, city: str) -> GeocodeResult:
        ...

    def fetch_weather(self, lat: float, lon: float) -> RawWeatherData:
        ...

    def compute_facts(
        self,
        raw_data: RawWeatherData | Dict[str, Any],
        window_name: str,
        now_local: Optional[datetime.datetime] = None,
    ) -> ComputedFactsResult:
        ...


def sanitize_location(location: str) -> str:
    """Sanitize user location input: strip, cap at 100 characters.

    Args:
        location: Raw user location text.

    Returns:
        Cleaned location string.

    Raises:
        LocationError: If empty or whitespace-only after cleaning.
    """
    if not location:
        raise LocationError("Location cannot be empty.")

    cleaned = location.strip()
    if len(cleaned) > 100:
        cleaned = cleaned[:100].strip()

    if not cleaned:
        raise LocationError("Location cannot be empty after sanitization.")

    return cleaned


class OpenMeteoClient:
    """Production Open-Meteo client utilizing sync httpx."""

    GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(
        self,
        facts_registry: Optional[FactsRegistry] = None,
        timeout_seconds: float = 10.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.facts_registry = facts_registry or FactsRegistry()
        self.timeout_seconds = timeout_seconds
        self._http_client = http_client

    def _get_client(self) -> httpx.Client:
        if self._http_client is not None:
            return self._http_client
        return httpx.Client(timeout=self.timeout_seconds)

    def geocode(self, city: str) -> GeocodeResult:
        """Resolve city name to coordinates and formatted place name.

        Args:
            city: City name provided by user.

        Returns:
            GeocodeResult containing lat, lon, resolved_name, and timezone.

        Raises:
            LocationError: On empty input, no results, HTTP error, timeout, or bad JSON.
        """
        clean_city = sanitize_location(city)
        params = {"name": clean_city, "count": 5, "format": "json"}

        client = self._get_client()
        try:
            # 10s timeout, 1 retry on timeout
            try:
                response = client.get(self.GEOCODE_URL, params=params)
            except (httpx.TimeoutException, httpx.ConnectTimeout):
                response = client.get(self.GEOCODE_URL, params=params)

            if response.status_code != 200:
                raise LocationError(
                    f"Geocoding service returned HTTP {response.status_code} for '{clean_city}'."
                )

            data = response.json()
        except LocationError:
            raise
        except (httpx.RequestError, ValueError) as err:
            raise LocationError(f"Geocoding request failed: {err}") from err

        results = data.get("results")
        if not results or not isinstance(results, list):
            raise LocationError(f"Could not find coordinates for location '{clean_city}'.")

        top = results[0]
        try:
            lat = float(top["latitude"])
            lon = float(top["longitude"])
            name = str(top.get("name", clean_city))
            admin1 = top.get("admin1")
            country = top.get("country")
            timezone = top.get("timezone", "UTC")

            parts = [name]
            if admin1 and admin1 != name:
                parts.append(str(admin1))
            if country:
                parts.append(str(country))
            resolved_name = ", ".join(parts)

            return GeocodeResult(
                lat=lat,
                lon=lon,
                resolved_name=resolved_name,
                timezone=timezone,
                country=country,
                admin1=admin1,
            )
        except (KeyError, TypeError, ValueError) as err:
            raise LocationError(f"Malformed geocoding response data: {err}") from err

    def fetch_weather(self, lat: float, lon: float) -> RawWeatherData:
        """Fetch fresh weather forecast from Open-Meteo using variables from facts.yaml.

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            RawWeatherData containing payload and fetch_time.

        Raises:
            WeatherAPIError: On timeout, HTTP error, or malformed data.
        """
        query_vars = self.facts_registry.get_query_variables()
        params: Dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "timezone": "auto",
            "forecast_days": 3,
        }

        if query_vars["current"]:
            params["current"] = ",".join(query_vars["current"])
        if query_vars["hourly"]:
            params["hourly"] = ",".join(query_vars["hourly"])
        if query_vars["daily"]:
            params["daily"] = ",".join(query_vars["daily"])

        client = self._get_client()
        try:
            try:
                response = client.get(self.FORECAST_URL, params=params)
            except (httpx.TimeoutException, httpx.ConnectTimeout):
                response = client.get(self.FORECAST_URL, params=params)

            if response.status_code != 200:
                raise WeatherAPIError(
                    f"Weather API returned HTTP {response.status_code}: {response.text[:200]}"
                )

            payload = response.json()
        except WeatherAPIError:
            raise
        except (httpx.RequestError, ValueError) as err:
            raise WeatherAPIError(f"Weather API request failed: {err}") from err

        fetch_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return RawWeatherData(payload=payload, fetch_time=fetch_time)

    def compute_facts(
        self,
        raw_data: RawWeatherData | Dict[str, Any],
        window_name: str,
        now_local: Optional[datetime.datetime] = None,
    ) -> ComputedFactsResult:
        """Aggregate raw meteorological data into domain facts for the requested window.

        Args:
            raw_data: Raw payload or RawWeatherData object.
            window_name: Target window name ('now', 'today', 'this_evening', 'tomorrow').
            now_local: Optional explicit local datetime. If None, derived strictly from API's utc_offset_seconds.

        Returns:
            ComputedFactsResult with evaluated facts and window metadata.

        Raises:
            WindowPassedError: If the requested window has fully passed.
        """
        payload = raw_data.payload if isinstance(raw_data, RawWeatherData) else raw_data
        fetch_time = (
            raw_data.fetch_time
            if isinstance(raw_data, RawWeatherData)
            else datetime.datetime.now(datetime.timezone.utc).isoformat()
        )

        utc_offset_seconds = int(payload.get("utc_offset_seconds", 0))
        loc_tz = datetime.timezone(datetime.timedelta(seconds=utc_offset_seconds))
        timezone_name = payload.get("timezone", "UTC")

        # Derive now_local from API utc_offset_seconds if not explicitly provided
        if now_local is None:
            now_local = datetime.datetime.now(datetime.timezone.utc).astimezone(loc_tz)
        elif now_local.tzinfo is None:
            now_local = now_local.replace(tzinfo=loc_tz)

        # Check base window
        effective_start, effective_end, is_partly_passed, is_fully_passed = (
            self.facts_registry.resolve_window(window_name, now_local)
        )
        start_str = effective_start.strftime("%H:%M")
        end_str = "midnight" if (effective_end.hour == 0 and effective_end.minute == 0) else effective_end.strftime("%H:%M")

        if is_fully_passed:
            raise WindowPassedError(
                f"The requested window '{window_name}' has already ended at {end_str} local time."
            )

        window_desc_map = {
            "today": "rest of today",
            "this_evening": "this evening",
            "tomorrow": "tomorrow daytime",
            "now": "next 3 hours",
        }
        desc = window_desc_map.get(window_name, window_name.replace("_", " "))
        window_note = f"Covers {start_str} to {end_str} ({desc})"

        # Hourly timestamps from API
        hourly_data = payload.get("hourly", {})
        hourly_times_raw = hourly_data.get("time", [])
        hourly_datetimes: List[datetime.datetime] = []
        for t_str in hourly_times_raw:
            try:
                # Open-Meteo returns ISO local times e.g. "2026-10-02T13:00"
                dt = datetime.datetime.fromisoformat(t_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=loc_tz)
                hourly_datetimes.append(dt)
            except ValueError:
                continue

        computed_facts: Dict[str, Optional[float | int | str]] = {}

        for fact in self.facts_registry.config.facts:
            # Fact-specific window override (e.g. UV 11:00-16:00)
            if fact.fixed_window is not None:
                f_start, f_end, f_partly, f_fully = self.facts_registry.resolve_window(
                    window_name, now_local, fixed_window=fact.fixed_window
                )
                if f_fully:
                    computed_facts[fact.name] = None
                    continue
                fact_start = f_start
                fact_end = f_end
            else:
                fact_start = effective_start
                fact_end = effective_end

            if fact.source == "current":
                current_section = payload.get("current", {})
                raw_val = current_section.get(fact.open_meteo_var)
                computed_facts[fact.name] = raw_val

            elif fact.source == "hourly":
                var_values = hourly_data.get(fact.open_meteo_var, [])
                window_values: List[float] = []

                for dt, val in zip(hourly_datetimes, var_values):
                    # Start inclusive, end exclusive
                    if fact_start <= dt < fact_end:
                        if val is not None:
                            try:
                                window_values.append(float(val))
                            except (ValueError, TypeError):
                                pass

                if not window_values:
                    computed_facts[fact.name] = None
                else:
                    computed_facts[fact.name] = self._aggregate(window_values, fact.agg)

            elif fact.source == "daily":
                daily_data = payload.get("daily", {})
                var_values = daily_data.get(fact.open_meteo_var, [])
                if var_values and var_values[0] is not None:
                    try:
                        computed_facts[fact.name] = float(var_values[0])
                    except (ValueError, TypeError):
                        computed_facts[fact.name] = None
                else:
                    computed_facts[fact.name] = None

        # Round every numeric fact to 1 decimal place (single source of truth)
        for k, v in computed_facts.items():
            if v is not None and isinstance(v, (int, float)) and not isinstance(v, bool):
                computed_facts[k] = round(float(v), 1)

        return ComputedFactsResult(
            facts=computed_facts,
            window_name=window_name,
            partly_passed=is_partly_passed,
            window_note=window_note,
            window_start=start_str,
            window_end=end_str,
            fetch_time=fetch_time,
            timezone=timezone_name,
            utc_offset_seconds=utc_offset_seconds,
        )

    @staticmethod
    def _aggregate(values: List[float], agg_type: str) -> Optional[Any]:
        """Aggregate non-empty float list per agg_type."""
        if not values:
            return None
        if agg_type == "set":
            unique_vals = set()
            for v in values:
                try:
                    unique_vals.add(int(v) if float(v).is_integer() else v)
                except (ValueError, TypeError):
                    unique_vals.add(v)
            return sorted(list(unique_vals))
        elif agg_type == "max":
            return round(max(values), 1)
        elif agg_type == "min":
            return round(min(values), 1)
        elif agg_type == "sum":
            return round(sum(values), 1)
        elif agg_type == "mean":
            return round(sum(values) / len(values), 1)
        elif agg_type == "current":
            return round(values[0], 1)
        return round(values[0], 1)
