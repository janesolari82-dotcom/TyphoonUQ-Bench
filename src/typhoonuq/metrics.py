"""Point and interval metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .constants import AGENCIES


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def mean_prediction_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


def in_range_rate(y_pred: np.ndarray, target_lower: np.ndarray, target_upper: np.ndarray) -> float:
    in_range = (y_pred >= target_lower) & (y_pred <= target_upper)
    return float(np.mean(in_range))


def distance_to_range(y_pred: np.ndarray, target_lower: np.ndarray, target_upper: np.ndarray) -> float:
    lower_gap = np.maximum(target_lower - y_pred, 0.0)
    upper_gap = np.maximum(y_pred - target_upper, 0.0)
    return float(np.mean(lower_gap + upper_gap))


def range_coverage_at_90(
    lower_90: np.ndarray,
    upper_90: np.ndarray,
    target_lower: np.ndarray,
    target_upper: np.ndarray,
) -> float:
    covered = (lower_90 <= target_lower) & (upper_90 >= target_upper)
    return float(np.mean(covered))


def calibration_error(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float = 0.8,
) -> float:
    coverage = interval_coverage(y_true, lower, upper)
    return abs(coverage - target_coverage)


def _core_metrics(
    prediction_df: pd.DataFrame,
    target_coverage: float = 0.8,
) -> dict[str, float]:
    y_true = prediction_df["target"].to_numpy(dtype=float)
    y_pred = prediction_df["prediction"].to_numpy(dtype=float)
    lower = prediction_df["lower"].to_numpy(dtype=float)
    upper = prediction_df["upper"].to_numpy(dtype=float)
    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "interval_coverage": interval_coverage(y_true, lower, upper),
        "interval_width": mean_prediction_interval_width(lower, upper),
        "calibration_error": calibration_error(y_true, lower, upper, target_coverage=target_coverage),
    }
    if {"target_lower", "target_upper"}.issubset(prediction_df.columns):
        range_df = prediction_df.dropna(subset=["prediction", "target_lower", "target_upper"]).copy()
        if not range_df.empty:
            range_pred = range_df["prediction"].to_numpy(dtype=float)
            target_lower = range_df["target_lower"].to_numpy(dtype=float)
            target_upper = range_df["target_upper"].to_numpy(dtype=float)
            metrics["in_range_rate"] = in_range_rate(range_pred, target_lower, target_upper)
            metrics["distance_to_range"] = distance_to_range(range_pred, target_lower, target_upper)
            if {"lower_90", "upper_90"}.issubset(range_df.columns):
                coverage_df = range_df.dropna(subset=["lower_90", "upper_90"])
                if not coverage_df.empty:
                    metrics["range_coverage_at_90"] = range_coverage_at_90(
                        coverage_df["lower_90"].to_numpy(dtype=float),
                        coverage_df["upper_90"].to_numpy(dtype=float),
                        coverage_df["target_lower"].to_numpy(dtype=float),
                        coverage_df["target_upper"].to_numpy(dtype=float),
                    )
    if "pressure_range" in prediction_df.columns:
        dispersion_df = prediction_df.dropna(subset=["pressure_range"]).copy()
        if not dispersion_df.empty:
            widths = dispersion_df["upper"].to_numpy(dtype=float) - dispersion_df["lower"].to_numpy(dtype=float)
            dispersion = dispersion_df["pressure_range"].to_numpy(dtype=float)
            if len(widths) > 1 and np.std(dispersion) > 0 and np.std(widths) > 0:
                metrics["dispersion_interval_corr"] = float(np.corrcoef(dispersion, widths)[0, 1])
            else:
                metrics["dispersion_interval_corr"] = 0.0
    return metrics


def _high_disagreement_breakdown(
    prediction_df: pd.DataFrame,
    target_coverage: float = 0.8,
) -> dict[str, Any] | None:
    if "pressure_range" not in prediction_df.columns:
        return None
    subset = prediction_df.dropna(subset=["pressure_range"]).copy()
    if subset.empty:
        return None
    threshold = float(subset["pressure_range"].quantile(0.75))
    subset = subset[subset["pressure_range"] >= threshold].copy()
    if subset.empty:
        return None
    metrics = _core_metrics(subset, target_coverage=target_coverage)
    metrics["count"] = int(len(subset))
    return metrics


def _single_agency_controls(prediction_df: pd.DataFrame) -> dict[str, Any] | None:
    per_agency: dict[str, dict[str, float]] = {}
    for agency in AGENCIES:
        column = f"pressure_{agency}"
        if column not in prediction_df.columns:
            continue
        subset = prediction_df.dropna(subset=["prediction", column]).copy()
        if subset.empty:
            continue
        y_true = subset[column].to_numpy(dtype=float)
        y_pred = subset["prediction"].to_numpy(dtype=float)
        per_agency[agency] = {
            "count": int(len(subset)),
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
        }
    if not per_agency:
        return None
    agency_maes = [metrics["mae"] for metrics in per_agency.values()]
    agency_rmses = [metrics["rmse"] for metrics in per_agency.values()]
    return {
        "agencies": per_agency,
        "mean_mae": float(np.mean(agency_maes)),
        "mean_rmse": float(np.mean(agency_rmses)),
    }


def evaluate_predictions(
    prediction_df: pd.DataFrame,
    target_coverage: float = 0.8,
) -> dict[str, Any]:
    required = ["target", "prediction", "lower", "upper"]
    missing = [column for column in required if column not in prediction_df.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    metrics = _core_metrics(prediction_df, target_coverage=target_coverage)
    breakdowns: dict[str, Any] = {}
    high_disagreement = _high_disagreement_breakdown(prediction_df, target_coverage=target_coverage)
    if high_disagreement is not None:
        breakdowns["high_disagreement"] = high_disagreement
    single_agency_controls = _single_agency_controls(prediction_df)
    if single_agency_controls is not None:
        breakdowns["single_agency_controls"] = single_agency_controls
    metrics["breakdowns"] = breakdowns
    return metrics
