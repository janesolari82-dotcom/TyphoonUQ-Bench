"""Small utility helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_utc_timestamp(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series, utc=True, errors="coerce")
    if hasattr(values.dt, "tz_convert"):
        values = values.dt.tz_convert(None)
    return values


def normalize_name(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    return "".join(ch for ch in text if ch.isalnum())


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def haversine_distance_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in angular degrees."""

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1 - a)))
    return math.degrees(c)


def timestamp_from_parts(df: pd.DataFrame, year_col: str, month_col: str, day_col: str, hour_col: str) -> pd.Series:
    return ensure_utc_timestamp(
        pd.to_datetime(
            {
                "year": pd.to_numeric(df[year_col], errors="coerce"),
                "month": pd.to_numeric(df[month_col], errors="coerce"),
                "day": pd.to_numeric(df[day_col], errors="coerce"),
                "hour": pd.to_numeric(df[hour_col], errors="coerce"),
            },
            errors="coerce",
            utc=True,
        )
    )
