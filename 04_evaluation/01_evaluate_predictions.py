"""Evaluate a saved prediction file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typhoonuq.io import load_table
from typhoonuq.metrics import evaluate_predictions
from typhoonuq.utils import dump_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--target-coverage", type=float, default=0.8)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    prediction_df = load_table(args.predictions)
    metrics = evaluate_predictions(prediction_df, target_coverage=args.target_coverage)
    if args.output_json:
        dump_json(metrics, args.output_json)
    print(metrics)


if __name__ == "__main__":
    main()
