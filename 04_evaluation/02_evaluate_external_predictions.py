"""Evaluate external model predictions by joining them to a task sample table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typhoonuq.io import load_table, save_table
from typhoonuq.metrics import evaluate_predictions
from typhoonuq.utils import dump_json


def _prediction_columns(df: pd.DataFrame) -> list[str]:
    optional = ["lower", "upper", "lower_90", "upper_90"]
    columns = ["sample_id", "prediction"]
    columns.extend(column for column in optional if column in df.columns)
    return columns


def _sample_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "sample_id",
        "target",
        "target_lower",
        "target_upper",
        "pressure_range",
        "agency_count",
        "split",
        "storm_id",
        "timestamp",
        "task",
        "pressure_jma",
        "pressure_jtwc",
        "pressure_cma",
        "pressure_hko",
    ]
    return [column for column in preferred if column in df.columns]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--target-coverage", type=float, default=0.8)
    parser.add_argument("--output-joined", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    samples = load_table(args.samples)
    predictions = load_table(args.predictions)

    missing = {"sample_id", "prediction"} - set(predictions.columns)
    if missing:
        raise SystemExit(f"External predictions are missing required columns: {sorted(missing)}")

    joined = samples[_sample_columns(samples)].merge(
        predictions[_prediction_columns(predictions)],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        raise SystemExit("No rows matched between samples and predictions by sample_id.")

    if args.split != "all":
        joined = joined[joined["split"] == args.split].copy()
    if joined.empty:
        raise SystemExit(f"No matched rows remain for split={args.split!r}.")

    if "lower" not in joined.columns:
        joined["lower"] = joined["prediction"]
    if "upper" not in joined.columns:
        joined["upper"] = joined["prediction"]

    metrics = evaluate_predictions(joined, target_coverage=args.target_coverage)
    metrics["evaluated_rows"] = int(len(joined))
    metrics["split"] = args.split

    if args.output_joined is not None:
        save_table(joined, args.output_joined)
    if args.output_json is not None:
        dump_json(metrics, args.output_json)
    print(metrics)


if __name__ == "__main__":
    main()
