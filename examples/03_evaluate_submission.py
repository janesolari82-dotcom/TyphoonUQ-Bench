"""Validate that a third-party ``predictions.parquet`` matches the benchmark schema.

This is a self-contained check that adapter authors can run *before* invoking
``04_evaluation/01_evaluate_predictions.py``. It does not score the predictions; it
verifies that every required column is present, has the expected dtype, and
contains no impossible values (e.g. ``lower > upper`` or missing ``sample_id``).

Usage:

    python examples/evaluate_submission.py \\
        --predictions outputs/external/my_model/analysis-0h/predictions.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from typhoonuq.io import load_table


REQUIRED_COLUMNS = {
    "sample_id":  "string",
    "target":     "float",
    "prediction": "float",
    "lower":      "float",
    "upper":      "float",
    "split":      "string",
}

OPTIONAL_COLUMNS = (
    "lower_90",
    "upper_90",
    "backbone",
    "device",
    "storm_id",
    "timestamp",
    "task",
    "target_lower",
    "target_upper",
    "pressure_range",
    "agency_count",
)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_stringlike(series: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def validate(predictions_path: Path) -> int:
    df = load_table(predictions_path)
    print(f"[INFO] loaded {len(df)} rows from {predictions_path}")

    errors: list[str] = []
    warnings: list[str] = []

    for col, expected in REQUIRED_COLUMNS.items():
        if col not in df.columns:
            errors.append(f"missing required column: '{col}'")
            continue
        if expected == "float" and not _is_numeric(df[col]):
            errors.append(f"column '{col}' must be numeric, got dtype={df[col].dtype}")
        if expected == "string" and not _is_stringlike(df[col]):
            errors.append(f"column '{col}' must be string-like, got dtype={df[col].dtype}")

    if "sample_id" in df.columns and df["sample_id"].isna().any():
        errors.append("'sample_id' contains NaN values")

    if "split" in df.columns:
        bad = sorted(set(df["split"].dropna()) - {"train", "val", "test"})
        if bad:
            errors.append(f"'split' contains unexpected values: {bad}")

    if {"lower", "upper"}.issubset(df.columns):
        bad_interval = (df["lower"] > df["upper"]).sum()
        if bad_interval:
            errors.append(f"{bad_interval} rows have lower > upper")

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            warnings.append(f"optional column '{col}' is not present")

    print()
    if warnings:
        print("[WARN] missing optional columns (the evaluator will still run):")
        for w in warnings:
            print(f"   - {w}")
        print()

    if errors:
        print("[FAIL] schema check failed:")
        for e in errors:
            print(f"   - {e}")
        return 1

    print("[OK] predictions.parquet conforms to the TyphoonUQ-Bench schema.")
    print("     You can now run 04_evaluation/01_evaluate_predictions.py on this file.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(validate(args.predictions))


if __name__ == "__main__":
    main()
