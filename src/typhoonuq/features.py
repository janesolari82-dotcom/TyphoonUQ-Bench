"""Feature engineering and sample generation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .constants import AGENCIES, DEFAULT_FEATURE_COLUMNS, IMAGE_BACKENDS, TASK_TO_HORIZON


def add_motion_features(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.sort_values(["storm_id", "timestamp"]).copy()
    out["prev_lat"] = out.groupby("storm_id")["lat"].shift(1)
    out["prev_lon"] = out.groupby("storm_id")["lon"].shift(1)
    out["motion_u"] = out["lon"] - out["prev_lon"]
    out["motion_v"] = out["lat"] - out["prev_lat"]
    out["motion_speed"] = np.sqrt(out["motion_u"].fillna(0) ** 2 + out["motion_v"].fillna(0) ** 2)
    out["month_sin"] = np.sin(2 * np.pi * out["month"].fillna(1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"].fillna(1) / 12.0)
    out["quality_flag"] = pd.to_numeric(out["quality_flag"], errors="coerce").fillna(0.0)
    return out.drop(columns=["prev_lat", "prev_lon"])


def _history_feature_names(feature_cols: Iterable[str], history_hours: int) -> list[str]:
    names: list[str] = []
    for lag in range(history_hours - 1, -1, -1):
        for column in feature_cols:
            names.append(f"{column}_t-{lag}")
    return names


def _resolve_backend_column(df: pd.DataFrame, image_backend: str) -> str:
    if image_backend not in IMAGE_BACKENDS:
        raise ValueError(f"Unsupported image backend: {image_backend}")
    ref_column = f"{image_backend}_ref"
    if ref_column in df.columns:
        return ref_column
    if "image_ref" in df.columns:
        return "image_ref"
    raise ValueError(f"Manifest does not contain a path column for backend `{image_backend}`")


def create_tabular_samples(
    manifest: pd.DataFrame,
    split_df: pd.DataFrame | None = None,
    task: str = "analysis-0h",
    history_hours: int = 6,
    feature_cols: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
    image_backend: str = "png",
) -> pd.DataFrame:
    if task not in TASK_TO_HORIZON:
        raise ValueError(f"Unsupported task: {task}")
    horizon = TASK_TO_HORIZON[task]
    work = add_motion_features(manifest)
    ref_column = _resolve_backend_column(work, image_backend)
    if split_df is not None:
        work = work.merge(split_df[["storm_id", "timestamp", "split"]], on=["storm_id", "timestamp"], how="left")
    else:
        work["split"] = "train"

    sample_rows = []
    feature_names = _history_feature_names(feature_cols, history_hours)
    for storm_id, group in work.groupby("storm_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        for idx in range(history_hours - 1, len(group) - horizon):
            history = group.iloc[idx - history_hours + 1 : idx + 1]
            target_row = group.iloc[idx + horizon]
            if pd.isna(target_row["pressure_consensus_median"]):
                continue
            if history[ref_column].isna().any():
                continue
            features = history.loc[:, feature_cols].fillna(0.0).to_numpy(dtype=float).reshape(-1)
            row = {
                "sample_id": f"{storm_id}_{target_row['timestamp'].strftime('%Y%m%d%H')}_{task}_{image_backend}",
                "storm_id": storm_id,
                "task": task,
                "timestamp": target_row["timestamp"],
                "split": target_row["split"],
                "image_backend": image_backend,
                "target": float(target_row["pressure_consensus_median"]),
                "target_lower": float(target_row["pressure_min"]) if pd.notna(target_row["pressure_min"]) else np.nan,
                "target_upper": float(target_row["pressure_max"]) if pd.notna(target_row["pressure_max"]) else np.nan,
                "pressure_range": float(target_row["pressure_range"]) if pd.notna(target_row["pressure_range"]) else np.nan,
                "agency_count": int(target_row["agency_count"]) if pd.notna(target_row["agency_count"]) else 0,
                "image_refs": "|".join(history[ref_column].astype(str).tolist()),
                "png_refs": "|".join(history["png_ref"].fillna("").astype(str).tolist()) if "png_ref" in history.columns else "",
                "h5_refs": "|".join(history["h5_ref"].fillna("").astype(str).tolist()) if "h5_ref" in history.columns else "",
            }
            for agency in AGENCIES:
                pressure_column = f"pressure_{agency}"
                row[pressure_column] = (
                    float(target_row[pressure_column])
                    if pressure_column in target_row.index and pd.notna(target_row[pressure_column])
                    else np.nan
                )
            row.update(dict(zip(feature_names, features)))
            sample_rows.append(row)
    return pd.DataFrame(sample_rows)
