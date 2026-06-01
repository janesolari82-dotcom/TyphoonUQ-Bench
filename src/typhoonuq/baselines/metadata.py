"""Metadata-only baseline and evaluation helpers."""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from typhoonuq.conformal import apply_symmetric_interval, fit_residual_quantile
from typhoonuq.constants import AGENCIES
from typhoonuq.metrics import evaluate_predictions

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None


@dataclass
class MetadataBaselineArtifacts:
    model_name: str
    feature_columns: list[str]
    metrics_by_split: dict[str, dict[str, Any]]
    residual_quantile: float
    predictions: pd.DataFrame


def _resolve_model(model_name: str = "auto"):
    if model_name in {"auto", "xgboost"} and XGBRegressor is not None:
        return "xgboost", XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        )
    return "gradient_boosting", GradientBoostingRegressor(random_state=42)


def train_metadata_baseline(
    samples: pd.DataFrame,
    model_name: str = "auto",
    target_coverage: float = 0.8,
) -> MetadataBaselineArtifacts:
    feature_columns = [column for column in samples.columns if "_t-" in column]
    if not feature_columns:
        raise ValueError("No tabular features were found in the sample table.")
    train_df = samples[samples["split"] == "train"].copy()
    val_df = samples[samples["split"] == "val"].copy()
    test_df = samples[samples["split"] == "test"].copy()
    resolved_name, model = _resolve_model(model_name)
    model.fit(train_df[feature_columns], train_df["target"])
    calibration_df = val_df if not val_df.empty else train_df
    calibration_pred = model.predict(calibration_df[feature_columns])
    residual_quantile = fit_residual_quantile(
        calibration_df["target"].to_numpy(dtype=float),
        calibration_pred,
        target_coverage=target_coverage,
    )
    residual_quantile_90 = fit_residual_quantile(
        calibration_df["target"].to_numpy(dtype=float),
        calibration_pred,
        target_coverage=0.9,
    )
    frames = []
    metrics_by_split: dict[str, dict[str, Any]] = {}
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if split_df.empty:
            continue
        preds = model.predict(split_df[feature_columns])
        lower, upper = apply_symmetric_interval(preds, residual_quantile)
        lower_90, upper_90 = apply_symmetric_interval(preds, residual_quantile_90)
        frame_columns = [
            "sample_id",
            "storm_id",
            "timestamp",
            "task",
            "target",
            "target_lower",
            "target_upper",
            "pressure_range",
            "agency_count",
            "split",
        ]
        frame_columns.extend([f"pressure_{agency}" for agency in AGENCIES])
        frame = split_df[
            frame_columns
        ].copy()
        frame["prediction"] = preds
        frame["lower"] = lower
        frame["upper"] = upper
        frame["lower_90"] = lower_90
        frame["upper_90"] = upper_90
        frames.append(frame)
        metrics_by_split[split_name] = evaluate_predictions(frame, target_coverage=target_coverage)
    prediction_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return MetadataBaselineArtifacts(
        model_name=resolved_name,
        feature_columns=feature_columns,
        metrics_by_split=metrics_by_split,
        residual_quantile=float(residual_quantile),
        predictions=prediction_df,
    )
