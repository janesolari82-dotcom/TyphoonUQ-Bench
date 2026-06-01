"""Dataset helpers for torch training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .constants import DEFAULT_FEATURE_COLUMNS, IMAGE_BACKENDS
from .features import create_tabular_samples
from .io import load_manifest, load_table

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency
    h5py = None

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    Dataset = object


def _resolve_file_path(ref: str, data_root: str | Path | None = None) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    if data_root is not None:
        return Path(data_root) / path
    return path


def _load_png_array(image_path: Path, image_size: int, allow_missing: bool = True) -> np.ndarray:
    if allow_missing and (not str(image_path) or str(image_path) == "nan" or not image_path.exists()):
        return np.zeros((1, image_size, image_size), dtype=np.float32)
    with Image.open(image_path) as img:
        img = img.convert("L").resize((image_size, image_size))
        array = np.asarray(img, dtype=np.float32) / 255.0
        return array[None, :, :]


def _load_h5_array(h5_path: Path, image_size: int, allow_missing: bool = True) -> np.ndarray:
    if allow_missing and (not str(h5_path) or str(h5_path) == "nan" or not h5_path.exists()):
        return np.zeros((1, image_size, image_size), dtype=np.float32)
    if h5py is None:
        raise ImportError("h5py is required to load H5 image backends")
    with h5py.File(h5_path, "r") as f:
        if "Infrared" not in f:
            raise KeyError(f"Expected 'Infrared' dataset in {h5_path}")
        array = np.asarray(f["Infrared"], dtype=np.float32)
    array = np.clip((array - 180.0) / 150.0, 0.0, 1.0)
    image = Image.fromarray((array * 255.0).astype(np.uint8), mode="L").resize((image_size, image_size))
    out = np.asarray(image, dtype=np.float32) / 255.0
    return out[None, :, :]


class TyphoonClipDataset(Dataset):
    """Clip dataset backed by a sample table and an explicit image backend."""

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        split_df: pd.DataFrame | str | Path | None = None,
        task: str = "analysis-0h",
        history_hours: int = 6,
        image_size: int = 224,
        feature_cols: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
        split_name: str = "train",
        image_backend: str = "png",
        data_root: str | Path | None = None,
        samples: pd.DataFrame | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required to use TyphoonClipDataset")
        if image_backend not in IMAGE_BACKENDS:
            raise ValueError(f"Unsupported image backend: {image_backend}")
        self.image_backend = image_backend
        self.data_root = Path(data_root) if data_root is not None else None
        self.image_size = image_size
        manifest_df = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest.copy()
        split_data = load_table(split_df) if isinstance(split_df, (str, Path)) else split_df
        sample_df = (
            samples.copy()
            if samples is not None
            else create_tabular_samples(
                manifest=manifest_df,
                split_df=split_data,
                task=task,
                history_hours=history_hours,
                feature_cols=feature_cols,
                image_backend=image_backend,
            )
        )
        self.samples = sample_df[sample_df["split"] == split_name].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frame(self, ref: str) -> np.ndarray:
        path = _resolve_file_path(ref, self.data_root)
        if self.image_backend == "png":
            return _load_png_array(path, self.image_size)
        return _load_h5_array(path, self.image_size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples.iloc[index]
        image_stack = [self._load_frame(ref) for ref in row["image_refs"].split("|")]
        images = np.stack(image_stack, axis=0)
        feature_values = row.filter(regex="_t-").to_numpy(dtype=np.float32)
        target = np.float32(row["target"])
        lower = np.float32(row["target_lower"]) if pd.notna(row["target_lower"]) else np.float32(target)
        upper = np.float32(row["target_upper"]) if pd.notna(row["target_upper"]) else np.float32(target)
        return {
            "images": torch.tensor(images, dtype=torch.float32),
            "tabular": torch.tensor(feature_values, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "interval": torch.tensor([lower, upper], dtype=torch.float32),
            "sample_id": row["sample_id"],
        }
