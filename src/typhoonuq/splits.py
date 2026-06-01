"""Benchmark split helpers."""

from __future__ import annotations

import pandas as pd

from .constants import AGENCIES, DEFAULT_FORWARD_SPLIT


def build_forward_main_split(
    manifest: pd.DataFrame,
    split_config: dict[str, tuple[int, int]] | None = None,
) -> pd.DataFrame:
    config = split_config or DEFAULT_FORWARD_SPLIT
    out = manifest[["storm_id", "timestamp", "year"]].copy()
    split_year = manifest["season"] if "season" in manifest.columns else manifest["year"]
    split_year = pd.to_numeric(split_year, errors="coerce").fillna(manifest["year"]).astype(int)
    out["split_year"] = split_year
    out["split"] = "unused"
    for split_name, (year_start, year_end) in config.items():
        mask = out["split_year"].between(year_start, year_end)
        out.loc[mask, "split"] = split_name
    return out.drop(columns=["split_year"])


def build_leave_one_agency_out(
    manifest: pd.DataFrame,
    agencies: tuple[str, ...] = AGENCIES,
) -> pd.DataFrame:
    frames = []
    base = manifest[["storm_id", "timestamp", "year"]].copy()
    for agency in agencies:
        frame = base.copy()
        frame["held_out_agency"] = agency
        frame["split"] = "train"
        frame.loc[frame["year"] >= 2020, "split"] = "test"
        frame.loc[frame["year"].between(2016, 2019), "split"] = "val"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validate_no_storm_leakage(split_df: pd.DataFrame) -> None:
    grouped = split_df.groupby("storm_id")["split"].nunique()
    leaking = grouped[grouped > 1]
    if not leaking.empty:
        leaking_ids = leaking.index.tolist()[:10]
        raise ValueError(f"Storm leakage detected across splits: {leaking_ids}")
