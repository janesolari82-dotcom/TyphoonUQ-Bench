"""Build task sample tables from a benchmark manifest and a split file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typhoonuq.features import create_tabular_samples
from typhoonuq.io import load_manifest, load_table, save_table
from typhoonuq.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/benchmark/benchmark_manifest.parquet"))
    parser.add_argument("--split-file", type=Path, default=Path("outputs/splits/forward_main.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/splits/sample_cache"))
    parser.add_argument("--history-hours", type=int, default=6)
    parser.add_argument("--image-backend", choices=["png", "h5"], default="png")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["analysis-0h", "forecast-6h", "forecast-12h"],
        help="Task names to export.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    split_df = load_table(args.split_file)
    output_dir = ensure_dir(args.output_dir)

    for task in args.tasks:
        samples = create_tabular_samples(
            manifest,
            split_df=split_df,
            task=task,
            history_hours=args.history_hours,
            image_backend=args.image_backend,
        )
        safe_task = task.replace("/", "_")
        output_path = output_dir / f"{safe_task}_{args.image_backend}_h{args.history_hours}_samples.parquet"
        save_table(samples, output_path)
        print(f"[INFO] wrote {len(samples)} rows: {output_path}")


if __name__ == "__main__":
    main()
