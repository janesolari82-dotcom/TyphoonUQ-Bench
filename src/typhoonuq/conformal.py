"""Simple conformal calibration utilities."""

from __future__ import annotations

import numpy as np


def fit_residual_quantile(y_true: np.ndarray, y_pred: np.ndarray, target_coverage: float = 0.8) -> float:
    alpha = 1.0 - target_coverage
    residuals = np.abs(y_true - y_pred)
    return float(np.quantile(residuals, 1.0 - alpha))


def apply_symmetric_interval(y_pred: np.ndarray, residual_quantile: float) -> tuple[np.ndarray, np.ndarray]:
    lower = y_pred - residual_quantile
    upper = y_pred + residual_quantile
    return lower, upper
