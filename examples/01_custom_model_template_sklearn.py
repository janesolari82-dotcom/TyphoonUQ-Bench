"""Minimal external-model template for TyphoonUQ-Bench task sample tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def feature_columns(samples: pd.DataFrame) -> list[str]:
    return [column for column in samples.columns if "_t-" in column]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/custom_model_predictions.parquet"))
    args = parser.parse_args()

    samples = pd.read_parquet(args.samples)
    features = feature_columns(samples)
    train = samples[samples["split"] == "train"].copy()
    test = samples[samples["split"] == "test"].copy()

    if train.empty or test.empty:
        raise SystemExit("The sample table must contain non-empty train and test splits.")

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(train[features], train["target"])
    prediction = model.predict(test[features])

    residual = abs(train["target"] - model.predict(train[features])).quantile(0.9)
    output = pd.DataFrame(
        {
            "sample_id": test["sample_id"].to_numpy(),
            "prediction": prediction,
            "lower": prediction - residual,
            "upper": prediction + residual,
            "lower_90": prediction - residual,
            "upper_90": prediction + residual,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(f"[INFO] wrote {len(output)} predictions: {args.output}")


if __name__ == "__main__":
    main()
