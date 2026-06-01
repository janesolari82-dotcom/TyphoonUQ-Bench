"""Storm-level case study export for TyphoonUQ-Bench."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from typhoonuq.io import load_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/benchmark/benchmark_manifest.parquet"),
        help="Benchmark manifest parquet/csv path.",
    )
    parser.add_argument(
        "--storm-name",
        type=str,
        default="HAIYAN",
        help="Storm name filter, case-insensitive.",
    )
    parser.add_argument(
        "--storm-id",
        type=str,
        default=None,
        help="Optional exact storm_id filter. When set, only this storm_id is kept.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional predictions parquet/csv to overlay model output for the same storm.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for case-study exports.",
    )
    return parser.parse_args()


def filter_storm(df: pd.DataFrame, storm_name: str, storm_id: str | None = None) -> pd.DataFrame:
    subset = df.copy()
    if storm_id is not None:
        if "storm_id" not in subset.columns:
            raise SystemExit("storm_id filter requested but input table has no storm_id column")
        subset = subset[subset["storm_id"].astype(str).eq(str(storm_id))].copy()
        if subset.empty:
            raise SystemExit(f"No rows found for storm_id={storm_id!r}")
    else:
        subset = subset[subset["storm_name"].astype(str).str.upper().eq(storm_name.upper())].copy()
        if subset.empty:
            raise SystemExit(f"No rows found for storm_name={storm_name!r}")
    if "timestamp" in subset.columns:
        subset = subset.sort_values("timestamp")
    return subset


def enrich_predictions_with_manifest(pred_df: pd.DataFrame, manifest_df: pd.DataFrame) -> pd.DataFrame:
    if "storm_name" in pred_df.columns:
        return pred_df
    manifest_lookup_columns = [
        "sample_id",
        "storm_id",
        "storm_name",
        "timestamp",
        "pressure_consensus_median",
        "pressure_min",
        "pressure_max",
    ]
    manifest_lookup_columns = [column for column in manifest_lookup_columns if column in manifest_df.columns]

    if "sample_id" in pred_df.columns and "sample_id" in manifest_df.columns:
        manifest_lookup = manifest_df[manifest_lookup_columns].drop_duplicates(subset=["sample_id"])
        enriched = pred_df.merge(manifest_lookup, on="sample_id", how="left", suffixes=("", "_manifest"))
    elif {"storm_id", "timestamp"}.issubset(pred_df.columns) and {"storm_id", "timestamp", "storm_name"}.issubset(manifest_df.columns):
        dedupe_subset = [column for column in ["storm_id", "timestamp"] if column in manifest_df.columns]
        manifest_lookup = manifest_df[manifest_lookup_columns].drop_duplicates(subset=dedupe_subset)
        enriched = pred_df.merge(
            manifest_lookup,
            on=["storm_id", "timestamp"],
            how="left",
            suffixes=("", "_manifest"),
        )
    else:
        raise SystemExit("Predictions are missing storm_name and cannot be joined to manifest")

    for column in ["storm_id", "timestamp"]:
        manifest_column = f"{column}_manifest"
        if manifest_column in enriched.columns:
            if column not in enriched.columns:
                enriched[column] = enriched[manifest_column]
            else:
                enriched[column] = enriched[column].where(enriched[column].notna(), enriched[manifest_column])
            enriched = enriched.drop(columns=[manifest_column])

    return enriched


def save_manifest_slice(storm_df: pd.DataFrame, output_dir: Path, storm_name: str) -> Path:
    columns = [
        "sample_id",
        "storm_id",
        "storm_name",
        "timestamp",
        "pressure_jma",
        "pressure_jtwc",
        "pressure_cma",
        "pressure_hko",
        "pressure_consensus_median",
        "pressure_min",
        "pressure_max",
        "pressure_range",
    ]
    columns = [column for column in columns if column in storm_df.columns]
    out_path = output_dir / f"{storm_name.lower()}_manifest_slice.csv"
    storm_df[columns].to_csv(out_path, index=False)
    return out_path


def plot_agency_range(storm_df: pd.DataFrame, output_dir: Path, storm_name: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"[WARN] matplotlib unavailable, skip manifest plot: {exc}")
        return None

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(storm_df["timestamp"], storm_df["pressure_consensus_median"], label="consensus median")
    ax.fill_between(
        storm_df["timestamp"],
        storm_df["pressure_min"],
        storm_df["pressure_max"],
        alpha=0.25,
        label="agency range",
    )
    for column, label in [
        ("pressure_jma", "JMA"),
        ("pressure_jtwc", "JTWC"),
        ("pressure_cma", "CMA"),
        ("pressure_hko", "HKO"),
    ]:
        if column in storm_df.columns and storm_df[column].notna().any():
            ax.plot(storm_df["timestamp"], storm_df[column], linewidth=1.0, alpha=0.8, label=label)
    ax.set_title(f"{storm_name.upper()} pressure case study")
    ax.set_ylabel("Pressure")
    ax.legend(loc="best", ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = output_dir / f"{storm_name.lower()}_agency_range.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_prediction_slice(pred_df: pd.DataFrame, output_dir: Path, storm_name: str) -> Path:
    columns = [
        "sample_id",
        "storm_id",
        "storm_name",
        "timestamp",
        "split",
        "task",
        "prediction",
        "target",
        "lower",
        "upper",
        "lower_90",
        "upper_90",
        "pressure_consensus_median",
        "pressure_range",
        "backbone",
        "pretrained",
        "device",
    ]
    columns = [column for column in columns if column in pred_df.columns]
    out_path = output_dir / f"{storm_name.lower()}_prediction_slice.csv"
    pred_df[columns].to_csv(out_path, index=False)
    return out_path


def plot_predictions(pred_df: pd.DataFrame, output_dir: Path, storm_name: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"[WARN] matplotlib unavailable, skip prediction plot: {exc}")
        return None

    fig, ax = plt.subplots(figsize=(12, 4))
    if "target" in pred_df.columns:
        ax.plot(pred_df["timestamp"], pred_df["target"], label="target", linewidth=2)
    ax.plot(pred_df["timestamp"], pred_df["prediction"], label="prediction", linewidth=2)
    if {"lower", "upper"}.issubset(pred_df.columns):
        ax.fill_between(pred_df["timestamp"], pred_df["lower"], pred_df["upper"], alpha=0.2, label="interval")
    ax.set_title(f"{storm_name.upper()} model case study")
    ax.set_ylabel("Pressure")
    task_values = ", ".join(sorted(pred_df["task"].dropna().astype(str).unique())) if "task" in pred_df.columns else ""
    if task_values:
        ax.set_xlabel(f"task={task_values}")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = output_dir / f"{storm_name.lower()}_prediction_plot.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_table(args.manifest)
    storm_df = filter_storm(manifest, args.storm_name, args.storm_id)

    label = f"{args.storm_name.upper()} storm_id={args.storm_id}" if args.storm_id is not None else args.storm_name.upper()
    print(f"[INFO] Manifest rows for {label}: {len(storm_df)}")
    print(f"[INFO] Timestamp range: {storm_df['timestamp'].min()} -> {storm_df['timestamp'].max()}")

    manifest_csv = save_manifest_slice(storm_df, output_dir, args.storm_name)
    print(f"[INFO] Wrote manifest slice: {manifest_csv}")

    manifest_plot = plot_agency_range(storm_df, output_dir, args.storm_name)
    if manifest_plot is not None:
        print(f"[INFO] Wrote manifest plot: {manifest_plot}")

    if args.predictions is None:
        return

    predictions = load_table(args.predictions)
    predictions = enrich_predictions_with_manifest(predictions, manifest)
    pred_df = filter_storm(predictions, args.storm_name, args.storm_id)
    prediction_csv = save_prediction_slice(pred_df, output_dir, args.storm_name)
    print(f"[INFO] Wrote prediction slice: {prediction_csv}")

    prediction_plot = plot_predictions(pred_df, output_dir, args.storm_name)
    if prediction_plot is not None:
        print(f"[INFO] Wrote prediction plot: {prediction_plot}")


if __name__ == "__main__":
    main()
