"""Template for plugging a custom UQ model into TyphoonUQ-Bench.

This file is a skeleton. Replace the marked sections with your own model and
training code. After running, the produced ``predictions.parquet`` is scored
by ``04_evaluation/01_evaluate_predictions.py`` exactly like the built-in baselines
and the TC-QFormer adapter.

Run from the repository root:

    python examples/02_custom_model_template_torch.py \\
        --manifest      data/processed/benchmark/benchmark_manifest.parquet \\
        --split-file    outputs/splits/forward_main.parquet \\
        --data-root     data/raw/digital_typhoon/archive \\
        --task          analysis-0h \\
        --image-backend png \\
        --output-dir    outputs/external/my_model/analysis-0h

Then evaluate:

    python 04_evaluation/01_evaluate_predictions.py \\
        --predictions outputs/external/my_model/analysis-0h/predictions.parquet \\
        --output-json outputs/external/my_model/analysis-0h/benchmark_metrics.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Make the in-tree typhoonuq package importable when the user has not installed
# the wheel. Safe to delete once `pip install -e .` has been run.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from typhoonuq.datasets import TyphoonClipDataset
from typhoonuq.features import create_tabular_samples
from typhoonuq.io import load_manifest, load_table, save_table
from typhoonuq.utils import ensure_dir


# ----------------------------------------------------------------------------
# Replace the body of `predict_one_batch` with your own model. The function
# must return three numpy arrays of shape ``(batch_size,)``:
#   - point:  the central pressure point estimate (hPa)
#   - lower:  the lower bound of the prediction interval (hPa)
#   - upper:  the upper bound of the prediction interval (hPa)
# ----------------------------------------------------------------------------
def predict_one_batch(batch: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Toy implementation: predict the per-batch mean pressure with +/- 5 hPa.

    Parameters
    ----------
    batch
        A dict with at least the following keys (produced by
        ``TyphoonClipDataset``):
          - ``images``  : tensor of shape (B, T, 1, 224, 224)
          - ``tabular`` : tensor of shape (B, F)
          - ``target``  : tensor of shape (B,) -- consensus pressure in hPa
          - ``sample_id``: list of B strings
    """
    target = batch["target"].cpu().numpy().astype(float)
    point = np.full_like(target, fill_value=float(target.mean()))
    lower = point - 5.0
    upper = point + 5.0
    return point, lower, upper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task", type=str, default="analysis-0h")
    parser.add_argument("--history-hours", type=int, default=6)
    parser.add_argument("--image-backend", type=str, choices=["png", "h5"], default="png")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - explicit user-facing message
        raise SystemExit(
            "PyTorch is required to run this template. Install with `pip install -e .[ml]`."
        ) from exc

    output_dir = ensure_dir(args.output_dir)

    manifest = load_manifest(args.manifest)
    split_df = load_table(args.split_file)
    samples = create_tabular_samples(
        manifest=manifest,
        split_df=split_df,
        task=args.task,
        history_hours=args.history_hours,
        image_backend=args.image_backend,
    )

    rows: list[dict] = []
    for split_name in ("train", "val", "test"):
        dataset = TyphoonClipDataset(
            manifest=args.manifest,
            split_df=args.split_file,
            task=args.task,
            history_hours=args.history_hours,
            image_size=224,
            split_name=split_name,
            image_backend=args.image_backend,
            data_root=args.data_root,
            samples=samples,
        )
        if len(dataset) == 0:
            continue
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        for batch in loader:
            point, lower, upper = predict_one_batch(batch)
            for idx, sid in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "sample_id": sid,
                        "target": float(batch["target"][idx].cpu()),
                        "prediction": float(point[idx]),
                        "lower": float(lower[idx]),
                        "upper": float(upper[idx]),
                        "split": split_name,
                        "backbone": "custom_template",
                        "device": "cpu",
                    }
                )

    predictions = pd.DataFrame(rows)
    # Merge in the per-sample metadata that the benchmark evaluator uses for
    # range-aware metrics (target_lower / target_upper / pressure_range / ...).
    metadata_cols = [
        c
        for c in (
            "sample_id",
            "storm_id",
            "timestamp",
            "task",
            "target_lower",
            "target_upper",
            "pressure_range",
            "agency_count",
        )
        if c in samples.columns
    ]
    metadata = samples[metadata_cols].drop_duplicates(subset=["sample_id"])
    predictions = predictions.merge(metadata, on="sample_id", how="left")
    save_table(predictions, output_dir / "predictions.parquet")
    print(f"[INFO] wrote {len(predictions)} predictions to {output_dir / 'predictions.parquet'}")


if __name__ == "__main__":
    main()
