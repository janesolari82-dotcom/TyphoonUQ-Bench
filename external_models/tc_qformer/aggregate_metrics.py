"""Aggregate per-task TC-QFormer benchmark metrics into a single summary.

Reads each `<output-root>/<task>/benchmark_metrics.json` (produced by
`04_evaluation/01_evaluate_predictions.py`) and writes a unified `summary.json` plus a
`summary.csv` for quick comparison with the built-in baselines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from typhoonuq.utils import dump_json


PRIMARY_METRIC_KEYS = (
    "mae",
    "rmse",
    "interval_coverage",
    "interval_width",
    "calibration_error",
    "in_range_rate",
    "distance_to_range",
    "range_coverage_at_90",
    "dispersion_interval_corr",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()

    summary: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for task in args.tasks:
        metrics_path = args.output_root / task / "benchmark_metrics.json"
        if not metrics_path.exists():
            print(f"[WARN] missing metrics file: {metrics_path}")
            continue
        with metrics_path.open("r", encoding="utf-8") as fp:
            metrics = json.load(fp)
        summary[task] = metrics
        row: dict[str, object] = {"task": task}
        for key in PRIMARY_METRIC_KEYS:
            if key in metrics:
                row[key] = metrics[key]
        rows.append(row)

    dump_json(summary, args.output_root / "summary.json")
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(args.output_root / "summary.csv", index=False)
        print(df.to_string(index=False))
    else:
        print("[WARN] no metrics aggregated")


if __name__ == "__main__":
    main()
