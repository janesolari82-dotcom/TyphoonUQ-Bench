"""Train TC-QFormer on TyphoonUQ-Bench and emit predictions.parquet.

Reproduction script for Guo et al. 2026 ("Interval-based TC Intensity Forecasting
with Spatiotemporal Transformers", RS 18(7):1069).  The script reuses the
TyphoonUQ-Bench data loader (`TyphoonClipDataset`), trains TC-QFormer with the
paper's two-stage strategy (L1 median pretraining -> multi-quantile fine-tune),
and writes a `predictions.parquet` / `metrics.json` pair compatible with the
benchmark's evaluation entry point (`04_evaluation/01_evaluate_predictions.py`).

Run via torchrun for multi-NPU or `python` for single device.

Example (single NPU):

  python external_models/tc_qformer/train_tc_qformer.py \
    --manifest data/processed/benchmark/benchmark_manifest.parquet \
    --split-file outputs/splits/forward_main.parquet \
    --data-root data/raw/digital_typhoon/archive \
    --task analysis-0h \
    --image-backend png \
    --device npu \
    --output-dir outputs/external/tc_qformer/analysis-0h
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
# Add both the repo root (for `external_models.*` imports) and `src/` (for
# `typhoonuq.*` imports) so this script works when invoked directly via
# `python external_models/tc_qformer/train_tc_qformer.py ...` from any CWD.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

try:
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    torch_npu = None

from typhoonuq.conformal import apply_symmetric_interval, fit_residual_quantile
from typhoonuq.constants import AGENCIES
from typhoonuq.datasets import TyphoonClipDataset
from typhoonuq.features import create_tabular_samples
from typhoonuq.io import load_manifest, load_table, save_table
from typhoonuq.metrics import evaluate_predictions
from typhoonuq.utils import dump_json, ensure_dir

from external_models.tc_qformer.model import (
    DEFAULT_QUANTILES,
    TCQFormer,
    decode_quantile_outputs,
    median_l1_loss,
    pinball_loss,
)


def _resolve_device(requested: str) -> tuple[str, torch.device]:
    if requested == "auto":
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu", torch.device("npu:0")
        if torch.cuda.is_available():
            return "cuda", torch.device("cuda:0")
        return "cpu", torch.device("cpu")
    if requested == "npu":
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            raise SystemExit("Requested --device npu but torch.npu is unavailable.")
        return "npu", torch.device("npu:0")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("Requested --device cuda but torch.cuda is unavailable.")
        return "cuda", torch.device("cuda:0")
    return "cpu", torch.device("cpu")


def _build_loaders(args: argparse.Namespace, samples: pd.DataFrame) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split_name in ("train", "val", "test"):
        dataset = TyphoonClipDataset(
            manifest=args.manifest,
            split_df=args.split_file,
            task=args.task,
            history_hours=args.history_hours,
            image_size=args.image_size_input,
            split_name=split_name,
            image_backend=args.image_backend,
            data_root=args.data_root,
            samples=samples,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split_name == "train"),
            num_workers=args.num_workers,
            pin_memory=False,
        )
    return loaders


def _build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _train_one_epoch(
    loader: DataLoader,
    model: TCQFormer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stage: int,
    quantiles: tuple[float, ...],
    progress_every: int,
    epoch: int,
    max_batches: int | None,
) -> float:
    model.train()
    losses: list[float] = []
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["images"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(images, tabular)
        if stage == 1:
            loss = median_l1_loss(output.squeeze(-1) if output.dim() == 2 else output[..., 0], target)
        else:
            loss = pinball_loss(output, target, quantiles)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if progress_every > 0 and batch_idx % progress_every == 0:
            print(
                f"[INFO] stage={stage} epoch={epoch + 1} batch {batch_idx}/{len(loader)} loss={losses[-1]:.6f}",
                flush=True,
            )
        if max_batches is not None and batch_idx >= max_batches:
            break
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def _evaluate_loss(
    loader: DataLoader,
    model: TCQFormer,
    device: torch.device,
    stage: int,
    quantiles: tuple[float, ...],
    max_batches: int | None,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["images"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)
        output = model(images, tabular)
        if stage == 1:
            loss = median_l1_loss(output.squeeze(-1) if output.dim() == 2 else output[..., 0], target)
        else:
            loss = pinball_loss(output, target, quantiles)
        losses.append(float(loss.detach().cpu()))
        if max_batches is not None and batch_idx >= max_batches:
            break
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def _predict(
    loader: DataLoader,
    model: TCQFormer,
    device: torch.device,
    quantiles: tuple[float, ...],
    split_name: str,
    target_coverage: float,
    max_batches: int | None,
) -> list[dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["images"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)
        output = model(images, tabular)
        # output shape: (B, H, |Q|) at stage 2.
        if output.dim() == 2:
            # Should not happen at inference (we always use stage 2 here).
            point = output.squeeze(-1)
            lower = point
            upper = point
        else:
            point_h, lower_h, upper_h = decode_quantile_outputs(output, quantiles, target_coverage)
            point = point_h.squeeze(-1)
            lower = lower_h.squeeze(-1)
            upper = upper_h.squeeze(-1)
        for idx, sample_id in enumerate(batch["sample_id"]):
            rows.append(
                {
                    "sample_id": sample_id,
                    "target": float(target[idx].cpu()),
                    "prediction": float(point[idx].cpu()),
                    "lower": float(lower[idx].cpu()),
                    "upper": float(upper[idx].cpu()),
                    "split": split_name,
                }
            )
        if max_batches is not None and batch_idx >= max_batches:
            break
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproduce TC-QFormer on TyphoonUQ-Bench")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task", type=str, default="analysis-0h")
    parser.add_argument("--history-hours", type=int, default=6)
    parser.add_argument("--image-backend", type=str, choices=["png", "h5"], default="png")
    parser.add_argument(
        "--image-size-input",
        type=int,
        default=128,
        help="Resize TyphoonClipDataset frames to this side length (the model further resizes to its internal size).",
    )
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "npu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stage1-epochs", type=int, default=15, help="L1 median pretraining (paper default 15).")
    parser.add_argument("--stage2-epochs", type=int, default=35, help="Multi-quantile fine-tuning (paper default 35).")
    parser.add_argument("--stage1-lr", type=float, default=5e-6, help="OneCycle peak LR for stage 1 (paper).")
    parser.add_argument("--stage2-lr", type=float, default=3e-6, help="Fixed LR for stage 2 (paper).")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Paper uses 1e-2 with AdamW.")
    parser.add_argument("--target-coverage", type=float, default=0.8)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = ensure_dir(args.output_dir)
    device_type, device = _resolve_device(args.device)
    if device_type == "npu":
        torch.npu.set_device(device)
    print(f"[INFO] using device={device_type} ({device})", flush=True)

    manifest = load_manifest(args.manifest)
    split_df = load_table(args.split_file)
    print(
        f"[INFO] task={args.task} backend={args.image_backend} manifest_rows={len(manifest)} split_rows={len(split_df)}",
        flush=True,
    )
    samples = create_tabular_samples(
        manifest=manifest,
        split_df=split_df,
        task=args.task,
        history_hours=args.history_hours,
        image_backend=args.image_backend,
    )
    print(f"[INFO] tabular samples rows={len(samples)}", flush=True)

    loaders = _build_loaders(args, samples=samples)
    print(
        "[INFO] dataset sizes "
        f"train={len(loaders['train'].dataset)} "
        f"val={len(loaders['val'].dataset)} "
        f"test={len(loaders['test'].dataset)}",
        flush=True,
    )

    tabular_dim = int(loaders["train"].dataset.samples.filter(regex="_t-").shape[1])
    model = TCQFormer(
        scalar_dim=tabular_dim,
        num_frames=args.history_hours,
        image_size=128,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        quantiles=DEFAULT_QUANTILES,
        horizon=1,
    ).to(device)

    print(
        f"[INFO] TCQFormer params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
        f"tabular_dim={tabular_dim}",
        flush=True,
    )

    # ---- Stage 1: deterministic L1 pretraining ---------------------------------
    model.set_stage(1)
    optimizer = _build_optimizer(model, lr=args.stage1_lr, weight_decay=args.weight_decay)
    stage1_total_steps = max(1, args.stage1_epochs * max(1, len(loaders["train"])))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.stage1_lr,
        total_steps=stage1_total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    history: list[dict[str, float]] = []
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    for epoch in range(args.stage1_epochs):
        print(f"[INFO] stage 1 epoch {epoch + 1}/{args.stage1_epochs}", flush=True)
        train_loss = _train_one_epoch(
            loaders["train"], model, optimizer, device, stage=1,
            quantiles=DEFAULT_QUANTILES, progress_every=args.progress_every,
            epoch=epoch, max_batches=args.max_train_batches,
        )
        for _ in range(max(1, len(loaders["train"]))):
            try:
                scheduler.step()
            except Exception:  # OneCycle exhausted - safe to ignore
                break
        val_loss = _evaluate_loss(
            loaders["val"], model, device, stage=1,
            quantiles=DEFAULT_QUANTILES, max_batches=args.max_eval_batches,
        )
        history.append({"stage": 1, "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[INFO] stage 1 epoch {epoch + 1} train_l1={train_loss:.6f} val_l1={val_loss:.6f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    # ---- Stage 2: multi-quantile fine-tuning -----------------------------------
    model.set_stage(2)
    optimizer = _build_optimizer(model, lr=args.stage2_lr, weight_decay=args.weight_decay)
    best_pinball = float("inf")
    best_state = deepcopy(model.state_dict())
    for epoch in range(args.stage2_epochs):
        print(f"[INFO] stage 2 epoch {epoch + 1}/{args.stage2_epochs}", flush=True)
        train_loss = _train_one_epoch(
            loaders["train"], model, optimizer, device, stage=2,
            quantiles=DEFAULT_QUANTILES, progress_every=args.progress_every,
            epoch=epoch, max_batches=args.max_train_batches,
        )
        val_loss = _evaluate_loss(
            loaders["val"], model, device, stage=2,
            quantiles=DEFAULT_QUANTILES, max_batches=args.max_eval_batches,
        )
        history.append({"stage": 2, "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[INFO] stage 2 epoch {epoch + 1} train_pinball={train_loss:.6f} val_pinball={val_loss:.6f}", flush=True)
        if val_loss < best_pinball:
            best_pinball = val_loss
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    # ---- Inference + post-hoc 90% interval calibration -------------------------
    rows_by_split: dict[str, list[dict[str, object]]] = {}
    for split_name in ("train", "val", "test"):
        if len(loaders[split_name].dataset) == 0:
            rows_by_split[split_name] = []
            continue
        rows_by_split[split_name] = _predict(
            loaders[split_name], model, device, DEFAULT_QUANTILES,
            split_name=split_name, target_coverage=args.target_coverage,
            max_batches=args.max_eval_batches,
        )

    val_rows = rows_by_split.get("val", [])
    residual_quantile_90 = 0.0
    if val_rows:
        targets = np.array([row["target"] for row in val_rows], dtype=float)
        preds = np.array([row["prediction"] for row in val_rows], dtype=float)
        residual_quantile_90 = fit_residual_quantile(targets, preds, target_coverage=0.9)

    # Merge sample metadata so range-aware metrics fire downstream.
    sample_metadata_columns = [
        "sample_id",
        "storm_id",
        "timestamp",
        "task",
        "target_lower",
        "target_upper",
        "pressure_range",
        "agency_count",
    ] + [f"pressure_{agency}" for agency in AGENCIES]
    sample_metadata_columns = [c for c in sample_metadata_columns if c in samples.columns]
    metadata = samples[sample_metadata_columns].drop_duplicates(subset=["sample_id"])

    frames: list[dict[str, object]] = []
    for split_name, rows in rows_by_split.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        # 90% interval via symmetric residual conformal calibration around point.
        if residual_quantile_90 > 0:
            lower_90, upper_90 = apply_symmetric_interval(df["prediction"].to_numpy(dtype=float), residual_quantile_90)
            df["lower_90"] = lower_90
            df["upper_90"] = upper_90
        else:
            df["lower_90"] = df["lower"]
            df["upper_90"] = df["upper"]
        df["backbone"] = "tc_qformer"
        df["device"] = device_type
        df = df.merge(metadata, on="sample_id", how="left")
        frames.append(df)

    if not frames:
        predictions = pd.DataFrame()
        metrics_by_split: dict[str, dict[str, object]] = {}
    else:
        predictions = pd.concat(frames, ignore_index=True)
        metrics_by_split = {
            split: evaluate_predictions(
                predictions[predictions["split"] == split],
                target_coverage=args.target_coverage,
            )
            for split in predictions["split"].unique()
        }

    save_table(predictions, output_dir / "predictions.parquet")
    dump_json(
        {
            "model": "tc_qformer",
            "task": args.task,
            "device": device_type,
            "image_backend": args.image_backend,
            "history_hours": args.history_hours,
            "embed_dim": args.embed_dim,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "patch_size": args.patch_size,
            "stage1_epochs": args.stage1_epochs,
            "stage2_epochs": args.stage2_epochs,
            "stage1_lr": args.stage1_lr,
            "stage2_lr": args.stage2_lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "history": history,
            "target_coverage": args.target_coverage,
            "conformal_residual_quantile_90": residual_quantile_90,
            "metrics_by_split": metrics_by_split,
        },
        output_dir / "metrics.json",
    )
    print(f"[INFO] wrote predictions and metrics to {output_dir}", flush=True)
    print(json.dumps({"metrics_by_split": metrics_by_split}, indent=2, default=str))


if __name__ == "__main__":
    main()
