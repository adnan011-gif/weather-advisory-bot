"""Facts registry and window definitions loader.

Loads and validates config/facts.yaml, provides Open-Meteo variable lists,
and handles window time calculations in local timezone.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


SourceType = Literal["current", "hourly", "daily"]
AggType = Literal["current", "max", "min", "sum", "mean", "set"]


class FixedWindow(BaseModel):
    """Fixed hourly window for a specific fact (e.g., UV 11:00 to 16:00)."""
    start_hour: int = Field(ge=0, le=23)
    end_hour: int = Field(ge=0, le=24)

    @model_validator(mode="after")
    def validate_hours(self) -> FixedWindow:
        if self.start_hour >= self.end_hour:
            raise ValueError(f"start_hour ({self.start_hour}) must be < end_hour ({self.end_hour})")
        return self


class FactDefinition(BaseModel):
    """Specification of a single meteorological fact."""
    name: str
    source: SourceType
    open_meteo_var: str
    agg: AggType
    fixed_window: Optional[FixedWindow] = None
    unit: str = ""


class WindowDefinition(BaseModel):
    """Definition of a named time window."""
    type: Literal["relative_hours", "remainder_of_day", "fixed_hours", "next_day_hours"]
    start_offset_hours: Optional[int] = None
    end_offset_hours: Optional[int] = None
    start_hour: Optional[int] = None
    end_hour: Optional[int] = None


class FactsConfig(BaseModel):
    """Container for the validated facts registry configuration."""
    windows: Dict[str, WindowDefinition]
    facts: List[FactDefinition]

    @field_validator("facts")
    @classmethod
    def validate_unique_fact_names(cls, facts: List[FactDefinition]) -> List[FactDefinition]:
        seen = set()
        for fact in facts:
            if fact.name in seen:
                raise ValueError(f"Duplicate fact name '{fact.name}' in facts registry.")
            seen.add(fact.name)
        return facts


class FactsRegistry:
    """Registry managing fact definitions, Open-Meteo query params, and time windows."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Initialize and validate the facts registry from YAML.

        Args:
            config_path: Path to facts.yaml. Defaults to config/facts.yaml relative to repo root.
        """
        if config_path is None:
            # Resolve relative to project root
            base_dir = Path(__file__).resolve().parent.parent
            config_path = base_dir / "config" / "facts.yaml"
        self.config_path = Path(config_path)
        self.config = self._load_and_validate(self.config_path)

    @staticmethod
    def _load_and_validate(path: Path) -> FactsConfig:
        """Load YAML file and validate against Pydantic schema."""
        if not path.is_file():
            raise FileNotFoundError(f"Facts config file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        if not isinstance(raw_data, dict):
            raise ValueError(f"Invalid format in {path}: expected a YAML mapping.")

        return FactsConfig.model_validate(raw_data)

    def get_query_variables(self) -> Dict[str, List[str]]:
        """Extract unique Open-Meteo variables needed for current, hourly, and daily API requests.

        Returns:
            Dictionary mapping source category ('current', 'hourly', 'daily') to list of variable names.
        """
        result: Dict[str, List[str]] = {
            "current": [],
            "hourly": [],
            "daily": [],
        }
        for fact in self.config.facts:
            var_list = result[fact.source]
            if fact.open_meteo_var not in var_list:
                var_list.append(fact.open_meteo_var)
        return result

    def get_fact(self, name: str) -> Optional[FactDefinition]:
        """Retrieve a fact definition by name."""
        for fact in self.config.facts:
            if fact.name == name:
                return fact
        return None

    def resolve_window(
        self,
        window_name: str,
        now_local: datetime.datetime,
        fixed_window: Optional[FixedWindow] = None,
    ) -> Tuple[datetime.datetime, datetime.datetime, bool, bool]:
        """Compute window start, end, and elapsed status in local time.

        Args:
            window_name: One of the defined windows in facts.yaml (e.g. 'now', 'today', 'this_evening', 'tomorrow').
            now_local: Current local timestamp at the target location.
            fixed_window: Optional override window specific to a fact.

        Returns:
            Tuple of (effective_start, effective_end, is_partly_passed, is_fully_passed)

        Raises:
            KeyError: If window_name is unknown.
        """
        if window_name not in self.config.windows:
            raise KeyError(
                f"Unknown window '{window_name}'. Available: {list(self.config.windows.keys())}"
            )

        win_def = self.config.windows[window_name]
        date_today = now_local.date()
        date_tomorrow = date_today + datetime.timedelta(days=1)

        # Base window calculation
        if win_def.type == "relative_hours":
            start_off = win_def.start_offset_hours or 0
            end_off = win_def.end_offset_hours or 3
            curr_hour = now_local.replace(minute=0, second=0, microsecond=0)
            base_start = curr_hour + datetime.timedelta(hours=start_off)
            base_end = curr_hour + datetime.timedelta(hours=end_off)
        elif win_def.type == "remainder_of_day":
            base_start = datetime.datetime.combine(date_today, datetime.time(win_def.start_hour or 0, 0))
            base_end = datetime.datetime.combine(date_today, datetime.time(win_def.end_hour or 23, 59, 59))
        elif win_def.type == "fixed_hours":
            base_start = datetime.datetime.combine(date_today, datetime.time(win_def.start_hour or 17, 0))
            base_end = datetime.datetime.combine(date_today, datetime.time(win_def.end_hour or 21, 0))
        elif win_def.type == "next_day_hours":
            base_start = datetime.datetime.combine(date_tomorrow, datetime.time(win_def.start_hour or 6, 0))
            base_end = datetime.datetime.combine(date_tomorrow, datetime.time(win_def.end_hour or 22, 0))
        else:
            raise ValueError(f"Unsupported window type: {win_def.type}")

        # If a fact has its own fixed window, apply it to the target date
        if fixed_window is not None:
            target_date = base_start.date()
            end_time = datetime.time(fixed_window.end_hour, 0) if fixed_window.end_hour < 24 else datetime.time(23, 59, 59)
            base_start = datetime.datetime.combine(target_date, datetime.time(fixed_window.start_hour, 0))
            base_end = datetime.datetime.combine(target_date, end_time)

        # Ensure timezone match if now_local has tzinfo
        if now_local.tzinfo is not None:
            if base_start.tzinfo is None:
                base_start = base_start.replace(tzinfo=now_local.tzinfo)
            if base_end.tzinfo is None:
                base_end = base_end.replace(tzinfo=now_local.tzinfo)

        # Check elapsed status
        if now_local >= base_end:
            return base_start, base_end, False, True

        is_partly_passed = False
        effective_start = base_start
        if now_local > base_start:
            is_partly_passed = True
            effective_start = now_local

        return effective_start, base_end, is_partly_passed, False
