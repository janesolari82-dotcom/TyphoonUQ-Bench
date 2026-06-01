"""Adapter for an extracted Digital Typhoon archive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from typhoonuq.constants import DT_FRAME_INDEX_COLUMNS, DT_STORM_INDEX_COLUMNS
from typhoonuq.io import save_table
from typhoonuq.utils import ensure_dir, normalize_name, timestamp_from_parts


@dataclass
class DigitalTyphoonIndexBundle:
    storm_index: pd.DataFrame
    frame_index: pd.DataFrame
    summary: dict[str, int]


def _resolve_archive_structure(archive_root: str | Path) -> tuple[Path, Path, Path, Path]:
    archive_root = Path(archive_root)
    metadata_json = archive_root / "metadata.json"
    metadata_dir = archive_root / "metadata" / "metadata"
    png_root = archive_root / "image_png" / "image_png"
    h5_root = archive_root / "image" / "image"
    if not metadata_json.exists():
        raise FileNotFoundError(f"Missing metadata.json under {archive_root}")
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Missing per-storm metadata CSV directory under {archive_root}")
    if not png_root.exists():
        raise FileNotFoundError(f"Missing PNG root under {archive_root}")
    if not h5_root.exists():
        raise FileNotFoundError(f"Missing H5 root under {archive_root}")
    return metadata_json, metadata_dir, png_root, h5_root


def build_digital_typhoon_indexes(
    archive_root: str | Path,
    year_min: int | None = None,
    year_max: int | None = None,
    storm_limit: int | None = None,
) -> DigitalTyphoonIndexBundle:
    metadata_json, metadata_dir, png_root, h5_root = _resolve_archive_structure(archive_root)
    archive_root = Path(archive_root)
    metadata = pd.read_json(metadata_json, typ="series")
    storm_rows = []
    frame_rows = []

    for storm_id_raw, payload in metadata.items():
        storm_id = str(storm_id_raw)
        season = int(payload.get("season")) if payload.get("season") is not None else None
        if year_min is not None and season is not None and season < year_min:
            continue
        if year_max is not None and season is not None and season > year_max:
            continue
        storm_rows.append(
            {
                "storm_id": storm_id,
                "storm_name": payload.get("name"),
                "season": season if season is not None else pd.NA,
                "start": payload.get("start"),
                "end": payload.get("end"),
                "images": int(payload.get("images", 0)),
                "normalized_name": normalize_name(payload.get("name")),
            }
        )
        csv_path = metadata_dir / f"{storm_id}.csv"
        if not csv_path.exists():
            continue
        storm_df = pd.read_csv(csv_path)
        if storm_df.empty:
            continue
        storm_df["storm_id"] = storm_id
        storm_df["storm_name"] = payload.get("name")
        storm_df["season"] = payload.get("season")
        storm_df["timestamp"] = timestamp_from_parts(storm_df, "year", "month", "day", "hour")
        storm_df["pressure_dt"] = pd.to_numeric(storm_df.get("pressure"), errors="coerce")
        storm_df["wind_dt"] = pd.to_numeric(storm_df.get("wind"), errors="coerce")
        storm_df["grade"] = pd.to_numeric(storm_df.get("grade"), errors="coerce")
        storm_df["intp"] = pd.to_numeric(storm_df.get("intp"), errors="coerce")
        storm_df["lat"] = pd.to_numeric(storm_df.get("lat"), errors="coerce")
        storm_df["lon"] = pd.to_numeric(storm_df.get("lng"), errors="coerce")
        storm_df["h5_ref"] = storm_df["file_1"].map(lambda value: str(Path("image") / "image" / storm_id / str(value)))
        storm_df["png_ref"] = storm_df["file_1"].map(
            lambda value: str(Path("image_png") / "image_png" / storm_id / Path(str(value)).with_suffix(".png").name)
        )
        storm_df["has_h5"] = storm_df["h5_ref"].map(lambda value: (archive_root / value).exists())
        storm_df["has_png"] = storm_df["png_ref"].map(lambda value: (archive_root / value).exists())
        frame_rows.append(storm_df.loc[:, DT_FRAME_INDEX_COLUMNS])
        if storm_limit is not None and len(storm_rows) >= storm_limit:
            break

    storm_index = pd.DataFrame(storm_rows).loc[:, DT_STORM_INDEX_COLUMNS].sort_values(["season", "storm_id"]).reset_index(drop=True)
    frame_index = (
        pd.concat(frame_rows, ignore_index=True).sort_values(["season", "storm_id", "timestamp"]).reset_index(drop=True)
        if frame_rows
        else pd.DataFrame(columns=DT_FRAME_INDEX_COLUMNS)
    )
    summary = {
        "storms": int(len(storm_index)),
        "frames": int(len(frame_index)),
        "png_frames": int(frame_index["has_png"].sum()) if not frame_index.empty else 0,
        "h5_frames": int(frame_index["has_h5"].sum()) if not frame_index.empty else 0,
    }
    return DigitalTyphoonIndexBundle(storm_index=storm_index, frame_index=frame_index, summary=summary)


def save_digital_typhoon_indexes(bundle: DigitalTyphoonIndexBundle, output_dir: str | Path) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    storm_path = output_dir / "storm_index.parquet"
    frame_path = output_dir / "frame_index.parquet"
    summary_path = output_dir / "summary.csv"
    save_table(bundle.storm_index, storm_path)
    save_table(bundle.frame_index, frame_path)
    save_table(pd.DataFrame([bundle.summary]), summary_path)
    return {
        "storm_index": storm_path,
        "frame_index": frame_path,
        "summary": summary_path,
    }
