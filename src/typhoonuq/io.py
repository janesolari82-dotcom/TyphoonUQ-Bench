"""Input and output helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import ensure_utc_timestamp


def _normalize_loaded_manifest(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df["timestamp"] = ensure_utc_timestamp(df["timestamp"])
    return df


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported table format: {path}")


def load_manifest(path: str | Path) -> pd.DataFrame:
    return _normalize_loaded_manifest(load_table(path))


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    save_table(df, path)
