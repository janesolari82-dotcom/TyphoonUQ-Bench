"""Console entrypoints for TyphoonUQ-Bench."""

from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import nullcontext
import os
from pathlib import Path
import time

import pandas as pd

from typhoonuq.adapters.digital_typhoon import build_digital_typhoon_indexes, save_digital_typhoon_indexes
from typhoonuq.adapters.jma_best_track import load_jma_best_track
from typhoonuq.alignment import (
    build_anchored_storm_match_table,
    build_benchmark_manifest,
    build_dt_jma_match_table,
    build_jma_ibtracs_match_table,
    standardize_ibtracs,
)
from typhoonuq.baselines.metadata import train_metadata_baseline
from typhoonuq.conformal import apply_symmetric_interval, fit_residual_quantile
from typhoonuq.constants import AGENCIES
from typhoonuq.datasets import TyphoonClipDataset
from typhoonuq.features import create_tabular_samples
from typhoonuq.io import load_manifest, load_table, save_table
from typhoonuq.metrics import evaluate_predictions
from typhoonuq.splits import build_forward_main_split, build_leave_one_agency_out, validate_no_storm_leakage
from typhoonuq.utils import dump_json, ensure_dir, load_yaml

try:
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    dist = None
    DataLoader = None
    DistributedSampler = None

try:
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    torch_npu = None

from typhoonuq.baselines.torch_models import (
    ImageOnlyGRURegressor,
    MultimodalGRURegressor,
    decode_output,
    gaussian_nll,
    quantile_loss,
)


class _NullScaler:
    def scale(self, loss):
        return loss

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


if torch is not None:
    _no_grad = torch.no_grad

    def _autocast_context(device_type: str, enabled: bool):
        if not enabled:
            return nullcontext()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type=device_type, enabled=enabled)
        if device_type == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "amp"):
            return torch.npu.amp.autocast(enabled=enabled)
        return torch.cuda.amp.autocast(enabled=enabled)

    def _build_grad_scaler(device_type: str, enabled: bool):
        if not enabled:
            return _NullScaler()
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                return torch.amp.GradScaler(device_type, enabled=enabled)
            except TypeError:  # pragma: no cover - older torch fallback
                return torch.amp.GradScaler(enabled=enabled)
        if device_type == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "amp"):
            return torch.npu.amp.GradScaler(enabled=enabled)
        return torch.cuda.amp.GradScaler(enabled=enabled)
else:  # pragma: no cover - optional dependency

    def _no_grad():
        def decorator(func):
            return func

        return decorator

    def _autocast_context(device_type: str, enabled: bool):
        return nullcontext()

    def _build_grad_scaler(device_type: str, enabled: bool):
        return _NullScaler()


def _npu_available() -> bool:
    if torch is None or not hasattr(torch, "npu"):
        return False
    try:
        return bool(torch.npu.is_available())
    except Exception:
        return False


def _resolve_device_type(requested: str) -> str:
    if torch is None:
        return "cpu"
    if requested == "auto":
        if _npu_available():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "npu" and not _npu_available():
        raise SystemExit("Requested --device npu but torch.npu is unavailable in the current environment.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda but torch.cuda is unavailable in the current environment.")
    return requested


def _distributed_world() -> dict[str, int | bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_distributed": world_size > 1,
    }


def _distributed_backend(device_type: str) -> str:
    if device_type == "npu":
        return "hccl"
    if device_type == "cuda":
        return "nccl"
    return "gloo"


def _is_main_process(world: dict[str, int | bool]) -> bool:
    return int(world["rank"]) == 0


def _set_device(device_type: str, local_rank: int) -> torch.device:
    if torch is None:
        raise SystemExit("torch is required for typhoonuq-train-torch")
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    if device_type == "npu":
        if not hasattr(torch, "npu"):
            raise SystemExit("torch.npu is unavailable; make sure the Ascend torch_npu stack is installed.")
        try:
            torch.npu.set_device(local_rank)
        except Exception:
            torch.npu.set_device(f"npu:{local_rank}")
        return torch.device(f"npu:{local_rank}")
    return torch.device("cpu")


def _init_distributed(device_type: str, world: dict[str, int | bool]) -> str | None:
    if not bool(world["is_distributed"]):
        return None
    if dist is None:
        raise SystemExit("torch.distributed is unavailable but torchrun/distributed environment variables were detected.")
    backend = _distributed_backend(device_type)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return backend


def _dist_ready() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()


def _all_reduce_mean(value: float, device: torch.device) -> float:
    if not _dist_ready():
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def _gather_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not _dist_ready():
        return rows
    gathered: list[list[dict[str, object]] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, rows)
    flat: list[dict[str, object]] = []
    for part in gathered:
        if part:
            flat.extend(part)
    return flat


def _barrier() -> None:
    if _dist_ready():
        dist.barrier()


def _cleanup_distributed() -> None:
    if _dist_ready():
        dist.destroy_process_group()


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _wrap_model(model, device_type: str, local_rank: int, is_distributed: bool):
    if not is_distributed:
        return model
    if device_type in {"cuda", "npu"}:
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    return torch.nn.parallel.DistributedDataParallel(model)


def _build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, distributed: bool, pin_memory: bool):
    sampler = None
    if distributed and DistributedSampler is not None and len(dataset) > 0:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, sampler


def build_index_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year-min", type=int)
    parser.add_argument("--year-max", type=int)
    parser.add_argument("--storm-limit", type=int)
    args = parser.parse_args(argv)
    bundle = build_digital_typhoon_indexes(
        args.archive_root,
        year_min=args.year_min,
        year_max=args.year_max,
        storm_limit=args.storm_limit,
    )
    paths = save_digital_typhoon_indexes(bundle, args.output_dir)
    print(bundle.summary)
    print(paths)


def build_manifest_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storm-index", type=Path, required=True)
    parser.add_argument("--frame-index", type=Path, required=True)
    parser.add_argument("--jma-best-track", type=Path, required=True)
    parser.add_argument("--ibtracs-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-columns", type=Path, default=Path("configs/source_columns.yaml"))
    parser.add_argument("--basin", type=str, default="WP")
    parser.add_argument("--year-min", type=int, default=1988)
    parser.add_argument("--year-max", type=int, default=2023)
    args = parser.parse_args(argv)

    mapping = load_yaml(args.source_columns)
    storm_index = load_table(args.storm_index)
    frame_index = load_table(args.frame_index)
    jma_df = load_jma_best_track(args.jma_best_track)
    ib_df = load_table(args.ibtracs_csv)
    standardized_ib = standardize_ibtracs(ib_df, mapping["ibtracs"])
    dt_jma_match_table = build_dt_jma_match_table(
        storm_index=storm_index,
        frame_index=frame_index,
        jma_df=jma_df,
        year_min=args.year_min,
        year_max=args.year_max,
    )
    jma_ibtracs_match_table = build_jma_ibtracs_match_table(
        jma_df=jma_df,
        ibtracs_df=standardized_ib,
        basin=args.basin.upper(),
        year_min=args.year_min,
        year_max=args.year_max,
    )
    match_table = build_anchored_storm_match_table(
        dt_jma_match_table=dt_jma_match_table,
        jma_ibtracs_match_table=jma_ibtracs_match_table,
    )
    manifest, coverage = build_benchmark_manifest(
        frame_index=frame_index,
        storm_match_table=match_table,
        ibtracs_df=standardized_ib,
        jma_df=jma_df,
        basin=args.basin.upper(),
        year_min=args.year_min,
        year_max=args.year_max,
    )
    output_dir = ensure_dir(args.output_dir)
    save_table(jma_df, output_dir / "jma_track.parquet")
    save_table(dt_jma_match_table, output_dir / "dt_jma_match_table.parquet")
    save_table(jma_ibtracs_match_table, output_dir / "jma_ibtracs_match_table.parquet")
    save_table(match_table, output_dir / "storm_match_table.parquet")
    save_table(match_table[match_table["status"] != "accepted"], output_dir / "manual_review.csv")
    save_table(manifest, output_dir / "benchmark_manifest.parquet")
    save_table(coverage, output_dir / "coverage_report.parquet")
    print(f"Saved benchmark artifacts under {output_dir}")


def make_splits_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    output_dir = ensure_dir(args.output_dir)
    forward_main = build_forward_main_split(manifest)
    validate_no_storm_leakage(forward_main)
    loo = build_leave_one_agency_out(manifest)
    save_table(forward_main, output_dir / "forward_main.parquet")
    save_table(loo, output_dir / "leave_one_agency_out.parquet")
    print(f"Saved split files under {output_dir}")


def train_metadata_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--task", type=str, default="analysis-0h")
    parser.add_argument("--history-hours", type=int, default=6)
    parser.add_argument("--image-backend", type=str, default="png")
    parser.add_argument("--model-name", type=str, default="auto")
    parser.add_argument("--target-coverage", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    output_dir = ensure_dir(args.output_dir)
    manifest = load_manifest(args.manifest)
    split_df = load_table(args.split_file)
    samples = create_tabular_samples(
        manifest,
        split_df=split_df,
        task=args.task,
        history_hours=args.history_hours,
        image_backend=args.image_backend,
    )
    save_table(samples, output_dir / "samples.parquet")
    artifacts = train_metadata_baseline(samples, model_name=args.model_name, target_coverage=args.target_coverage)
    save_table(artifacts.predictions, output_dir / "predictions.parquet")
    dump_json(
        {
            "model_name": artifacts.model_name,
            "feature_columns": artifacts.feature_columns,
            "metrics_by_split": artifacts.metrics_by_split,
            "residual_quantile": artifacts.residual_quantile,
            "image_backend": args.image_backend,
        },
        output_dir / "metrics.json",
    )
    print(f"Saved metadata baseline outputs to {output_dir}")


def _build_model(
    mode: str,
    tabular_dim: int,
    probabilistic: str,
    backbone: str = "tiny",
    pretrained: bool = False,
    temporal_backend: str = "gru",
):
    if mode == "image-only":
        return ImageOnlyGRURegressor(
            probabilistic=probabilistic,
            backbone=backbone,
            pretrained=pretrained,
            temporal_backend=temporal_backend,
        )
    if mode == "multimodal":
        return MultimodalGRURegressor(
            tabular_dim=tabular_dim,
            probabilistic=probabilistic,
            backbone=backbone,
            pretrained=pretrained,
            temporal_backend=temporal_backend,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def _prepare_model_for_device(model, device_type: str):
    if torch is None:
        return model
    if (
        device_type == "npu"
        and getattr(model, "temporal_backend", None) == "gru"
        and hasattr(model, "temporal")
        and getattr(model, "temporal", None) is not None
    ):
        # Ascend GRU kernels in this stack require fp16 recurrent weights.
        model.temporal = model.temporal.half()
    return model


def _sample_cache_path(split_file: Path, task: str, image_backend: str, history_hours: int) -> Path:
    sample_cache_dir = split_file.parent / "sample_cache"
    sample_cache_dir.mkdir(parents=True, exist_ok=True)
    safe_task = task.replace("/", "_")
    return sample_cache_dir / f"{safe_task}_{image_backend}_h{history_hours}_samples.parquet"


def _load_or_build_shared_samples(
    manifest: pd.DataFrame,
    split_df: pd.DataFrame,
    args: argparse.Namespace,
    is_main_process: bool,
) -> pd.DataFrame:
    cache_path = _sample_cache_path(args.split_file, args.task, args.image_backend, args.history_hours)
    lock_dir = cache_path.with_suffix(cache_path.suffix + ".lock")

    if cache_path.exists():
        if is_main_process:
            print(f"[INFO] loading shared sample table from cache: {cache_path}", flush=True)
        return load_table(cache_path)

    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if cache_path.exists():
                if is_main_process:
                    print(f"[INFO] loading shared sample table from cache: {cache_path}", flush=True)
                return load_table(cache_path)
            if is_main_process:
                print(f"[INFO] waiting for sample cache lock: {lock_dir}", flush=True)
            time.sleep(5)

    try:
        if is_main_process:
            print("[INFO] building shared sample table", flush=True)
        samples = create_tabular_samples(
            manifest,
            split_df=split_df,
            task=args.task,
            history_hours=args.history_hours,
            image_backend=args.image_backend,
        )
        save_table(samples, cache_path)
        if is_main_process:
            print(f"[INFO] wrote shared sample cache: {cache_path}", flush=True)
        return samples
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass


def _compute_loss(output: torch.Tensor, target: torch.Tensor, probabilistic: str) -> torch.Tensor:
    if probabilistic == "gaussian":
        return gaussian_nll(output, target)
    if probabilistic == "quantile":
        return quantile_loss(output, target)
    raise ValueError(f"Unsupported probabilistic mode: {probabilistic}")


def _forward_model(model, images, tabular, mode: str):
    return model(images) if mode == "image-only" else model(images, tabular)


def _run_epoch(
    loader,
    sampler,
    model,
    optimizer,
    scaler,
    device,
    mode: str,
    probabilistic: str,
    use_amp: bool,
    epoch: int,
    is_main_process: bool,
    progress_every: int,
    max_batches: int | None,
) -> float:
    if sampler is not None:
        sampler.set_epoch(epoch)
    model.train()
    losses = []
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["images"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device.type, use_amp):
            output = _forward_model(model, images, tabular, mode)
            loss = _compute_loss(output, target, probabilistic)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        if is_main_process and progress_every > 0 and batch_idx % progress_every == 0:
            print(
                f"[INFO] epoch {epoch + 1} batch {batch_idx}/{len(loader)} "
                f"loss={losses[-1]:.6f}",
                flush=True,
            )
        if max_batches is not None and batch_idx >= max_batches:
            if is_main_process:
                print(f"[INFO] stopping train epoch early at batch {batch_idx}", flush=True)
            break
    return sum(losses) / max(1, len(losses))


@_no_grad()
def _evaluate_loss(loader, model, device, mode: str, probabilistic: str, use_amp: bool, max_batches: int | None) -> float:
    model.eval()
    losses = []
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["images"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)
        with _autocast_context(device.type, use_amp):
            output = _forward_model(model, images, tabular, mode)
            loss = _compute_loss(output, target, probabilistic)
        losses.append(float(loss.detach().cpu()))
        if max_batches is not None and batch_idx >= max_batches:
            break
    return sum(losses) / max(1, len(losses))


@_no_grad()
def _predict(loader, model, device, mode: str, probabilistic: str, split_name: str, use_amp: bool, max_batches: int | None):
    model.eval()
    rows = []
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["images"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)
        with _autocast_context(device.type, use_amp):
            output = _forward_model(model, images, tabular, mode)
        center, lower, upper = decode_output(output, mode=probabilistic)
        for idx, sample_id in enumerate(batch["sample_id"]):
            rows.append(
                {
                    "sample_id": sample_id,
                    "target": float(target[idx].cpu()),
                    "prediction": float(center[idx].cpu()),
                    "lower": float(lower[idx].cpu()),
                    "upper": float(upper[idx].cpu()),
                    "split": split_name,
                }
            )
        if max_batches is not None and batch_idx >= max_batches:
            break
    return rows


def train_torch_main(argv: list[str] | None = None) -> None:
    if torch is None or DataLoader is None:
        raise SystemExit("torch is required for typhoonuq-train-torch")
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--task", type=str, default="analysis-0h")
    parser.add_argument("--mode", type=str, choices=["image-only", "multimodal"], default="multimodal")
    parser.add_argument("--probabilistic", type=str, choices=["gaussian", "quantile"], default="gaussian")
    parser.add_argument("--history-hours", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-backend", type=str, choices=["png", "h5"], default="png")
    parser.add_argument("--backbone", type=str, choices=["tiny", "resnet18"], default="tiny")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "npu"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temporal-backend", type=str, choices=["auto", "gru", "mean", "tcn"], default="auto")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--target-coverage", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.pretrained and args.backbone != "resnet18":
        parser.error("--pretrained is only supported with --backbone resnet18")

    world = _distributed_world()
    device_type = _resolve_device_type(args.device)
    device = _set_device(device_type, int(world["local_rank"]) if bool(world["is_distributed"]) else 0)
    ddp_backend = _init_distributed(device_type, world)
    is_main_process = _is_main_process(world)
    # Keep AMP on CUDA, but keep the Ascend path in fp32 for stability.
    use_amp = device_type == "cuda"
    pin_memory = device_type == "cuda"
    early_stopping_patience = 2

    output_dir = ensure_dir(args.output_dir)
    manifest = load_manifest(args.manifest)
    split_df = load_table(args.split_file)
    temporal_backend = "mean" if args.temporal_backend == "auto" and device_type == "npu" else args.temporal_backend
    if temporal_backend == "auto":
        temporal_backend = "gru"
    if is_main_process:
        print(
            f"[INFO] task={args.task} mode={args.mode} backbone={args.backbone} "
            f"device={device_type} distributed={bool(world['is_distributed'])} "
            f"world_size={int(world['world_size'])}",
            flush=True,
        )
        print(
            f"[INFO] manifest={args.manifest} split_file={args.split_file} data_root={args.data_root}",
            flush=True,
        )
        print(f"[INFO] temporal_backend={temporal_backend}", flush=True)
    shared_samples = _load_or_build_shared_samples(manifest, split_df, args, is_main_process)
    if is_main_process:
        print(f"[INFO] shared sample table rows={len(shared_samples)}", flush=True)

    train_dataset = TyphoonClipDataset(
        manifest,
        split_df=split_df,
        task=args.task,
        history_hours=args.history_hours,
        image_size=args.image_size,
        split_name="train",
        image_backend=args.image_backend,
        data_root=args.data_root,
        samples=shared_samples,
    )
    val_dataset = TyphoonClipDataset(
        manifest,
        split_df=split_df,
        task=args.task,
        history_hours=args.history_hours,
        image_size=args.image_size,
        split_name="val",
        image_backend=args.image_backend,
        data_root=args.data_root,
        samples=shared_samples,
    )
    test_dataset = TyphoonClipDataset(
        manifest,
        split_df=split_df,
        task=args.task,
        history_hours=args.history_hours,
        image_size=args.image_size,
        split_name="test",
        image_backend=args.image_backend,
        data_root=args.data_root,
        samples=shared_samples,
    )
    train_loader, train_sampler = _build_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        distributed=bool(world["is_distributed"]),
        pin_memory=pin_memory,
    )
    val_loader, _ = _build_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        distributed=bool(world["is_distributed"]),
        pin_memory=pin_memory,
    )
    test_loader, _ = _build_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        distributed=bool(world["is_distributed"]),
        pin_memory=pin_memory,
    )
    if is_main_process:
        print(
            f"[INFO] dataset sizes train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} "
            f"batch_size={args.batch_size} num_workers={args.num_workers}",
            flush=True,
        )

    tabular_dim = int(train_dataset.samples.filter(regex="_t-").shape[1]) if not train_dataset.samples.empty else 0
    model = _build_model(
        args.mode,
        tabular_dim=tabular_dim,
        probabilistic=args.probabilistic,
        backbone=args.backbone,
        pretrained=args.pretrained,
        temporal_backend=temporal_backend,
    )
    model = _prepare_model_for_device(model, device_type=device_type).to(device)
    model = _wrap_model(model, device_type=device_type, local_rank=int(world["local_rank"]), is_distributed=bool(world["is_distributed"]))
    optimizer = torch.optim.AdamW(_unwrap_model(model).parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    scaler = _build_grad_scaler(device_type, use_amp)

    history = []
    best_state = deepcopy(_unwrap_model(model).state_dict())
    best_epoch = 0
    best_score = float("inf")
    best_val_loss = float("inf")
    stopped_early = False
    epochs_without_improvement = 0

    try:
        for epoch in range(args.epochs):
            if is_main_process:
                print(f"[INFO] epoch {epoch + 1}/{args.epochs} start", flush=True)
            train_loss_local = _run_epoch(
                train_loader,
                train_sampler,
                model,
                optimizer,
                scaler,
                device,
                args.mode,
                args.probabilistic,
                use_amp,
                epoch,
                is_main_process=is_main_process,
                progress_every=args.progress_every,
                max_batches=args.max_train_batches,
            )
            train_loss = _all_reduce_mean(train_loss_local, device)
            val_loss_local = (
                _evaluate_loss(
                    val_loader,
                    model,
                    device,
                    args.mode,
                    args.probabilistic,
                    use_amp,
                    max_batches=args.max_eval_batches,
                )
                if len(val_dataset)
                else train_loss_local
            )
            val_loss = _all_reduce_mean(val_loss_local, device) if len(val_dataset) else train_loss
            scheduler.step(val_loss)
            if is_main_process:
                history.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    }
                )
                print(
                    f"[INFO] epoch {epoch + 1}/{args.epochs} done "
                    f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                    f"lr={float(optimizer.param_groups[0]['lr']):.6g}",
                    flush=True,
                )
            reference_score = val_loss if val_loss == val_loss else train_loss
            if reference_score < best_score - 1e-8:
                best_score = reference_score
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = deepcopy(_unwrap_model(model).state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    stopped_early = True
                    break

        if best_state is not None:
            _unwrap_model(model).load_state_dict(best_state)

        frames = []
        val_rows = (
            _predict(
                val_loader,
                model,
                device,
                args.mode,
                args.probabilistic,
                "val",
                use_amp,
                max_batches=args.max_eval_batches,
            )
            if len(val_dataset)
            else []
        )
        val_rows = _gather_rows(val_rows)
        residual_quantile = 0.0
        residual_quantile_90 = 0.0
        if val_rows:
            targets = torch.tensor([row["target"] for row in val_rows], dtype=torch.float32).numpy()
            preds = torch.tensor([row["prediction"] for row in val_rows], dtype=torch.float32).numpy()
            residual_quantile = fit_residual_quantile(targets, preds, target_coverage=args.target_coverage)
            residual_quantile_90 = fit_residual_quantile(targets, preds, target_coverage=0.9)

        for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            if len(loader.dataset) == 0:
                continue
            rows = _predict(
                loader,
                model,
                device,
                args.mode,
                args.probabilistic,
                split_name,
                use_amp,
                max_batches=args.max_eval_batches,
            )
            rows = _gather_rows(rows)
            if not is_main_process:
                continue
            for row in rows:
                lower, upper = apply_symmetric_interval(
                    torch.tensor([row["prediction"]]).numpy(),
                    residual_quantile,
                )
                row["lower"] = min(row["lower"], float(lower[0]))
                row["upper"] = max(row["upper"], float(upper[0]))
                lower_90, upper_90 = apply_symmetric_interval(
                    torch.tensor([row["prediction"]]).numpy(),
                    residual_quantile_90,
                )
                row["lower_90"] = float(lower_90[0])
                row["upper_90"] = float(upper_90[0])
            frames.extend(rows)

        if is_main_process:
            if frames:
                predictions = pd.DataFrame(frames)
                predictions["backbone"] = args.backbone
                predictions["pretrained"] = bool(args.pretrained)
                predictions["device"] = device_type
                predictions["distributed"] = bool(world["is_distributed"])
                predictions["world_size"] = int(world["world_size"])
                sample_metadata_columns = [
                    "sample_id",
                    "storm_id",
                    "timestamp",
                    "task",
                    "target_lower",
                    "target_upper",
                    "pressure_range",
                    "agency_count",
                ]
                sample_metadata_columns.extend([f"pressure_{agency}" for agency in AGENCIES])
                sample_metadata = pd.concat(
                    [
                        train_dataset.samples[sample_metadata_columns],
                        val_dataset.samples[sample_metadata_columns],
                        test_dataset.samples[sample_metadata_columns],
                    ],
                    ignore_index=True,
                )
                predictions = predictions.merge(sample_metadata, on="sample_id", how="left")
                save_table(predictions, output_dir / "predictions.parquet")
                metrics_by_split = {
                    split: evaluate_predictions(predictions[predictions["split"] == split], target_coverage=args.target_coverage)
                    for split in predictions["split"].unique()
                }
            else:
                predictions = pd.DataFrame()
                save_table(predictions, output_dir / "predictions.parquet")
                metrics_by_split = {}

            dump_json(
                {
                    "mode": args.mode,
                    "probabilistic": args.probabilistic,
                    "backbone": args.backbone,
                    "pretrained": bool(args.pretrained),
                    "temporal_backend": temporal_backend,
                    "device": device_type,
                    "requested_device": args.device,
                    "distributed": bool(world["is_distributed"]),
                    "world_size": int(world["world_size"]),
                    "ddp_backend": ddp_backend,
                    "epochs": args.epochs,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "stopped_early": stopped_early,
                    "early_stopping_patience": early_stopping_patience,
                    "amp_enabled": use_amp,
                    "num_workers": args.num_workers,
                    "history": history,
                    "target_coverage": args.target_coverage,
                    "conformal_residual_quantile": residual_quantile,
                    "conformal_residual_quantile_90": residual_quantile_90,
                    "metrics_by_split": metrics_by_split,
                    "image_backend": args.image_backend,
                    "data_root": str(args.data_root) if args.data_root else None,
                },
                output_dir / "metrics.json",
            )
            print(f"Saved torch baseline outputs to {output_dir}")
    finally:
        _barrier()
        _cleanup_distributed()
