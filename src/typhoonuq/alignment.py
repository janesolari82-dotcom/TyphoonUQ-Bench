"""Source normalization, storm matching, and benchmark manifest construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import AGENCIES
from .utils import ensure_utc_timestamp, haversine_distance_deg, normalize_name


def _rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    lower_lookup = {column.lower(): column for column in df.columns}
    keep = {}
    for target, source in mapping.items():
        if source in df.columns:
            keep[source] = target
            continue
        actual = lower_lookup.get(str(source).lower())
        if actual is not None:
            keep[actual] = target
    return df.rename(columns=keep).copy()


def standardize_digital_typhoon(
    df: pd.DataFrame,
    mapping: dict[str, Any],
    image_root: str | Path | None = None,
) -> pd.DataFrame:
    out = _rename_columns(df, mapping)
    for column in [
        "storm_id",
        "storm_name",
        "season",
        "basin",
        "timestamp",
        "image_ref",
        "quality_flag",
        "lat",
        "lon",
    ]:
        if column not in out.columns:
            out[column] = pd.NA
    out["timestamp"] = ensure_utc_timestamp(out["timestamp"])
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["quality_flag"] = pd.to_numeric(out["quality_flag"], errors="coerce").fillna(0.0)
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["storm_name_key"] = out["storm_name"].map(normalize_name)
    if image_root:
        root = Path(image_root)
        out["image_ref"] = out["image_ref"].map(lambda value: str(root / str(value)) if pd.notna(value) else value)
    return out


def standardize_ibtracs(df: pd.DataFrame, mapping: dict[str, Any]) -> pd.DataFrame:
    base_mapping = {k: v for k, v in mapping.items() if isinstance(v, str)}
    out = _rename_columns(df, base_mapping)
    lower_lookup = {column.lower(): column for column in df.columns}
    for column in ["storm_id", "storm_name", "season", "basin", "timestamp", "lat", "lon"]:
        if column not in out.columns:
            out[column] = pd.NA
    if "timestamp" in out.columns:
        raw_timestamp = out["timestamp"].astype(str)
        out = out[raw_timestamp.str.contains(r"\d{4}-\d{2}-\d{2}", regex=True, na=False)].copy()
    out["timestamp"] = ensure_utc_timestamp(out["timestamp"])
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["storm_name_key"] = out["storm_name"].map(normalize_name)
    for agency, source in mapping.get("pressure", {}).items():
        actual = source if source in df.columns else lower_lookup.get(str(source).lower())
        out[f"pressure_{agency}"] = pd.to_numeric(df[actual], errors="coerce") if actual is not None else np.nan
    for agency, source in mapping.get("wind", {}).items():
        actual = source if source in df.columns else lower_lookup.get(str(source).lower())
        out[f"wind_{agency}"] = pd.to_numeric(df[actual], errors="coerce") if actual is not None else np.nan
    for agency, value in mapping.get("wind_avg_period", {}).items():
        out[f"wind_avg_period_{agency}"] = value
    return out


def _derive_consensus_columns(
    manifest: pd.DataFrame,
    agencies: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pressure_columns = [f"pressure_{agency}" for agency in agencies]
    coverage = pd.DataFrame(
        {
            "agency": list(agencies),
            "coverage": [float(manifest[column].notna().mean()) for column in pressure_columns],
        }
    )
    selected = [agency for agency in agencies if coverage.loc[coverage["agency"] == agency, "coverage"].item() >= 0.25]
    if not selected:
        selected = list(agencies)
    use_columns = [f"pressure_{agency}" for agency in selected]
    pressure_frame = manifest[use_columns]
    manifest["pressure_consensus_median"] = pressure_frame.median(axis=1, skipna=True)
    manifest["pressure_min"] = pressure_frame.min(axis=1, skipna=True)
    manifest["pressure_max"] = pressure_frame.max(axis=1, skipna=True)
    manifest["pressure_range"] = manifest["pressure_max"] - manifest["pressure_min"]
    manifest["pressure_std"] = pressure_frame.std(axis=1, skipna=True, ddof=0)
    manifest["agency_count"] = pressure_frame.notna().sum(axis=1)
    manifest["metric_agencies"] = ",".join(selected)
    return manifest, coverage


def _compute_track_score(left_track: pd.DataFrame, right_track: pd.DataFrame) -> tuple[int, float]:
    merged = left_track.merge(right_track, on="timestamp", suffixes=("_left", "_right"))
    merged = merged.dropna(subset=["lat_left", "lon_left", "lat_right", "lon_right"])
    if merged.empty:
        return 0, float("inf")
    distances = [
        haversine_distance_deg(row.lat_left, row.lon_left, row.lat_right, row.lon_right)
        for row in merged.itertuples(index=False)
    ]
    return int(len(distances)), float(np.median(distances))


def _summarize_storms(
    df: pd.DataFrame,
    storm_id_col: str,
    storm_name_col: str,
    season_col: str = "season",
) -> pd.DataFrame:
    work = df[[storm_id_col, storm_name_col, season_col]].drop_duplicates().copy()
    work["storm_name_key"] = work[storm_name_col].map(normalize_name)
    return work.rename(columns={storm_id_col: "storm_id", storm_name_col: "storm_name", season_col: "season"})


def _track_lookup(df: pd.DataFrame, storm_id_col: str) -> dict[str, pd.DataFrame]:
    return {
        str(storm_id): group[["timestamp", "lat", "lon"]].dropna().copy()
        for storm_id, group in df.groupby(storm_id_col)
    }


def _build_storm_match_table_generic(
    left_storms: pd.DataFrame,
    left_tracks: dict[str, pd.DataFrame],
    right_storms: pd.DataFrame,
    right_tracks: dict[str, pd.DataFrame],
    left_out_id: str,
    left_out_name: str,
    right_out_id: str,
    right_out_name: str,
    min_overlap: int = 12,
    max_median_distance_deg: float = 1.0,
) -> pd.DataFrame:
    rows = []
    for row in left_storms.itertuples(index=False):
        season = int(row.season)
        normalized_name = normalize_name(row.storm_name)
        candidates = right_storms[(right_storms["season"] == season) & (right_storms["storm_name_key"] == normalized_name)]
        if normalized_name and len(candidates) == 1:
            candidate = candidates.iloc[0]
            rows.append(
                {
                    left_out_id: row.storm_id,
                    left_out_name: row.storm_name,
                    "season": season,
                    right_out_id: candidate["storm_id"],
                    right_out_name: candidate["storm_name"],
                    "match_method": "season+name",
                    "match_confidence": 1.0,
                    "overlap_steps": pd.NA,
                    "median_distance_deg": pd.NA,
                    "status": "accepted",
                }
            )
            continue

        if candidates.empty:
            candidates = right_storms[right_storms["season"] == season]

        best_match = None
        best_overlap = 0
        best_distance = float("inf")
        left_track = left_tracks.get(str(row.storm_id), pd.DataFrame(columns=["timestamp", "lat", "lon"]))
        for candidate in candidates.itertuples(index=False):
            right_track = right_tracks.get(str(candidate.storm_id), pd.DataFrame(columns=["timestamp", "lat", "lon"]))
            overlap_steps, median_distance = _compute_track_score(left_track, right_track)
            if overlap_steps > best_overlap or (overlap_steps == best_overlap and median_distance < best_distance):
                best_match = candidate
                best_overlap = overlap_steps
                best_distance = median_distance

        accepted = best_match is not None and best_overlap >= min_overlap and best_distance <= max_median_distance_deg
        confidence = (
            max(0.0, 1.0 - (best_distance / max_median_distance_deg)) * min(1.0, best_overlap / max(min_overlap, 1))
            if accepted
            else 0.0
        )
        rows.append(
            {
                left_out_id: row.storm_id,
                left_out_name: row.storm_name,
                "season": season,
                right_out_id: best_match.storm_id if best_match is not None else pd.NA,
                right_out_name: best_match.storm_name if best_match is not None else pd.NA,
                "match_method": "track" if accepted else "manual_review",
                "match_confidence": float(confidence),
                "overlap_steps": best_overlap if best_match is not None else 0,
                "median_distance_deg": best_distance if best_match is not None else pd.NA,
                "status": "accepted" if accepted else "manual_review",
            }
        )
    columns = [
        left_out_id,
        left_out_name,
        "season",
        right_out_id,
        right_out_name,
        "match_method",
        "match_confidence",
        "overlap_steps",
        "median_distance_deg",
        "status",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(["season", left_out_id]).reset_index(drop=True)


def build_storm_match_table(
    storm_index: pd.DataFrame,
    frame_index: pd.DataFrame,
    ibtracs_df: pd.DataFrame,
    basin: str = "WP",
    year_min: int = 1988,
    year_max: int = 2023,
    min_overlap: int = 12,
    max_median_distance_deg: float = 1.0,
) -> pd.DataFrame:
    dt_storms = storm_index.copy()
    dt_storms["storm_name_key"] = dt_storms["storm_name"].map(normalize_name)
    dt_storms = dt_storms[dt_storms["season"].between(year_min, year_max)]
    dt_storms = dt_storms.rename(columns={"storm_id": "storm_id", "storm_name": "storm_name"})
    ib_work = ibtracs_df.copy()
    ib_work = ib_work[ib_work["basin"].fillna("").astype(str).str.upper().eq(basin)]
    ib_work = ib_work[ib_work["season"].between(year_min, year_max)]
    ib_storms = _summarize_storms(ib_work, "storm_id", "storm_name")
    dt_tracks = _track_lookup(frame_index, "storm_id")
    ib_tracks = _track_lookup(ib_work, "storm_id")
    return _build_storm_match_table_generic(
        left_storms=dt_storms[["storm_id", "storm_name", "season", "storm_name_key"]],
        left_tracks=dt_tracks,
        right_storms=ib_storms,
        right_tracks=ib_tracks,
        left_out_id="dt_storm_id",
        left_out_name="dt_storm_name",
        right_out_id="ibtracs_sid",
        right_out_name="ibtracs_name",
        min_overlap=min_overlap,
        max_median_distance_deg=max_median_distance_deg,
    )


def build_dt_jma_match_table(
    storm_index: pd.DataFrame,
    frame_index: pd.DataFrame,
    jma_df: pd.DataFrame,
    year_min: int = 1988,
    year_max: int = 2023,
    min_overlap: int = 12,
    max_median_distance_deg: float = 1.0,
) -> pd.DataFrame:
    dt_storms = storm_index.copy()
    dt_storms["storm_name_key"] = dt_storms["storm_name"].map(normalize_name)
    dt_storms = dt_storms[dt_storms["season"].between(year_min, year_max)]
    jma_work = jma_df.copy()
    jma_work = jma_work[jma_work["season"].between(year_min, year_max)]
    jma_storms = _summarize_storms(jma_work, "jma_storm_id", "storm_name")
    dt_tracks = _track_lookup(frame_index, "storm_id")
    jma_tracks = _track_lookup(jma_work, "jma_storm_id")
    return _build_storm_match_table_generic(
        left_storms=dt_storms[["storm_id", "storm_name", "season", "storm_name_key"]],
        left_tracks=dt_tracks,
        right_storms=jma_storms,
        right_tracks=jma_tracks,
        left_out_id="dt_storm_id",
        left_out_name="dt_storm_name",
        right_out_id="jma_storm_id",
        right_out_name="jma_storm_name",
        min_overlap=min_overlap,
        max_median_distance_deg=max_median_distance_deg,
    )


def build_jma_ibtracs_match_table(
    jma_df: pd.DataFrame,
    ibtracs_df: pd.DataFrame,
    basin: str = "WP",
    year_min: int = 1988,
    year_max: int = 2023,
    min_overlap: int = 12,
    max_median_distance_deg: float = 1.0,
) -> pd.DataFrame:
    jma_work = jma_df.copy()
    jma_work = jma_work[jma_work["season"].between(year_min, year_max)]
    ib_work = ibtracs_df.copy()
    ib_work = ib_work[ib_work["basin"].fillna("").astype(str).str.upper().eq(basin)]
    ib_work = ib_work[ib_work["season"].between(year_min, year_max)]
    jma_storms = _summarize_storms(jma_work, "jma_storm_id", "storm_name")
    ib_storms = _summarize_storms(ib_work, "storm_id", "storm_name")
    jma_tracks = _track_lookup(jma_work, "jma_storm_id")
    ib_tracks = _track_lookup(ib_work, "storm_id")
    return _build_storm_match_table_generic(
        left_storms=jma_storms,
        left_tracks=jma_tracks,
        right_storms=ib_storms,
        right_tracks=ib_tracks,
        left_out_id="jma_storm_id",
        left_out_name="jma_storm_name",
        right_out_id="ibtracs_sid",
        right_out_name="ibtracs_name",
        min_overlap=min_overlap,
        max_median_distance_deg=max_median_distance_deg,
    )


def build_anchored_storm_match_table(
    dt_jma_match_table: pd.DataFrame,
    jma_ibtracs_match_table: pd.DataFrame,
) -> pd.DataFrame:
    dt_jma = dt_jma_match_table.copy().rename(
        columns={
            "match_method": "dt_jma_match_method",
            "match_confidence": "dt_jma_match_confidence",
            "status": "dt_jma_status",
        }
    )
    jma_ibtracs = jma_ibtracs_match_table.copy().rename(
        columns={
            "match_method": "jma_ibtracs_match_method",
            "match_confidence": "jma_ibtracs_match_confidence",
            "status": "jma_ibtracs_status",
        }
    )
    merged = dt_jma.merge(
        jma_ibtracs[
            [
                "season",
                "jma_storm_id",
                "jma_storm_name",
                "ibtracs_sid",
                "ibtracs_name",
                "jma_ibtracs_match_method",
                "jma_ibtracs_match_confidence",
                "jma_ibtracs_status",
            ]
        ],
        on=["season", "jma_storm_id", "jma_storm_name"],
        how="left",
    )
    rows = []
    for row in merged.itertuples(index=False):
        accepted = row.dt_jma_status == "accepted" and row.jma_ibtracs_status == "accepted" and pd.notna(row.ibtracs_sid)
        if row.dt_jma_status != "accepted":
            review_stage = "dt_to_jma"
        elif row.jma_ibtracs_status != "accepted" or pd.isna(row.ibtracs_sid):
            review_stage = "jma_to_ibtracs"
        else:
            review_stage = pd.NA
        rows.append(
            {
                "dt_storm_id": row.dt_storm_id,
                "dt_storm_name": row.dt_storm_name,
                "season": row.season,
                "jma_storm_id": row.jma_storm_id,
                "jma_storm_name": row.jma_storm_name,
                "ibtracs_sid": row.ibtracs_sid if accepted else pd.NA,
                "ibtracs_name": row.ibtracs_name if accepted else pd.NA,
                "match_method": "jma_anchor" if accepted else "manual_review",
                "match_confidence": min(float(row.dt_jma_match_confidence), float(row.jma_ibtracs_match_confidence)) if accepted else 0.0,
                "status": "accepted" if accepted else "manual_review",
                "review_stage": review_stage,
                "dt_jma_match_method": row.dt_jma_match_method,
                "dt_jma_match_confidence": row.dt_jma_match_confidence,
                "jma_ibtracs_match_method": row.jma_ibtracs_match_method,
                "jma_ibtracs_match_confidence": row.jma_ibtracs_match_confidence,
                "anchor_source": "jma_best_track",
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "dt_storm_id",
                "dt_storm_name",
                "season",
                "jma_storm_id",
                "jma_storm_name",
                "ibtracs_sid",
                "ibtracs_name",
                "match_method",
                "match_confidence",
                "status",
                "review_stage",
                "dt_jma_match_method",
                "dt_jma_match_confidence",
                "jma_ibtracs_match_method",
                "jma_ibtracs_match_confidence",
                "anchor_source",
            ]
        )
    return pd.DataFrame(rows).sort_values(["season", "dt_storm_id"]).reset_index(drop=True)


def _backend_available(has_png: Any, has_h5: Any) -> str:
    if bool(has_png) and bool(has_h5):
        return "png|h5"
    if bool(has_png):
        return "png"
    if bool(has_h5):
        return "h5"
    return ""


def build_benchmark_manifest(
    frame_index: pd.DataFrame,
    storm_match_table: pd.DataFrame,
    ibtracs_df: pd.DataFrame,
    jma_df: pd.DataFrame | None = None,
    agencies: tuple[str, ...] = AGENCIES,
    basin: str = "WP",
    year_min: int = 1988,
    year_max: int = 2023,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = storm_match_table[storm_match_table["status"] == "accepted"].copy()
    frame_work = frame_index.copy()
    frame_work = frame_work[frame_work["season"].between(year_min, year_max)]
    frame_work = frame_work.merge(
        accepted[
            [
                "dt_storm_id",
                "ibtracs_sid",
                "ibtracs_name",
                "jma_storm_id",
                "jma_storm_name",
                "match_method",
                "match_confidence",
                "dt_jma_match_method",
                "dt_jma_match_confidence",
                "jma_ibtracs_match_method",
                "jma_ibtracs_match_confidence",
                "anchor_source",
            ]
        ],
        left_on="storm_id",
        right_on="dt_storm_id",
        how="inner",
    )
    ib_work = ibtracs_df.copy()
    ib_work = ib_work[ib_work["basin"].fillna("").astype(str).str.upper().eq(basin)]
    ib_work = ib_work[ib_work["season"].between(year_min, year_max)]
    ib_work = ib_work.rename(columns={"storm_id": "ibtracs_sid", "storm_name": "ibtracs_name_src"})
    manifest = frame_work.merge(ib_work, on=["ibtracs_sid", "timestamp"], how="left", suffixes=("", "_ib"))
    if jma_df is not None and not jma_df.empty:
        jma_work = jma_df.copy()
        jma_work = jma_work[jma_work["season"].between(year_min, year_max)]
        manifest = manifest.merge(
            jma_work[
                [
                    "jma_storm_id",
                    "timestamp",
                    "pressure_jma_native",
                    "wind_jma_native",
                    "wind_avg_period_jma_native",
                    "jma_grade",
                    "landfall_flag",
                ]
            ],
            on=["jma_storm_id", "timestamp"],
            how="left",
        )
    manifest["storm_name"] = manifest["storm_name"].fillna(manifest["ibtracs_name"])
    manifest["quality_flag"] = (1.0 - manifest["intp"].fillna(1.0)).clip(lower=0.0, upper=1.0)
    manifest["image_backend_available"] = [
        _backend_available(has_png, has_h5) for has_png, has_h5 in zip(manifest["has_png"], manifest["has_h5"])
    ]
    manifest["image_ref"] = np.where(manifest["has_png"], manifest["png_ref"], manifest["h5_ref"])
    manifest["source_provenance"] = "digital_typhoon_archive+jma_best_track+ibtracs_v04r01"
    manifest["year"] = manifest["timestamp"].dt.year
    manifest["month"] = manifest["timestamp"].dt.month
    manifest["basin"] = basin
    manifest, coverage = _derive_consensus_columns(manifest, agencies)
    for agency in agencies:
        for column in [f"pressure_{agency}", f"wind_{agency}", f"wind_avg_period_{agency}"]:
            if column not in manifest.columns:
                manifest[column] = pd.NA
    ordered = [
        "storm_id",
        "storm_name",
        "season",
        "timestamp",
        "lat",
        "lon",
        "png_ref",
        "h5_ref",
        "image_ref",
        "image_backend_available",
        "quality_flag",
        "pressure_dt",
        "wind_dt",
        "grade",
        "intp",
        "jma_storm_id",
        "jma_storm_name",
        "pressure_jma_native",
        "wind_jma_native",
        "wind_avg_period_jma_native",
        "jma_grade",
        "landfall_flag",
        "ibtracs_sid",
    ]
    ordered.extend([f"pressure_{agency}" for agency in agencies])
    ordered.extend([f"wind_{agency}" for agency in agencies])
    ordered.extend([f"wind_avg_period_{agency}" for agency in agencies])
    ordered.extend(
        [
            "pressure_consensus_median",
            "pressure_min",
            "pressure_max",
            "pressure_range",
            "pressure_std",
            "agency_count",
            "metric_agencies",
            "match_method",
            "match_confidence",
            "dt_jma_match_method",
            "dt_jma_match_confidence",
            "jma_ibtracs_match_method",
            "jma_ibtracs_match_confidence",
            "anchor_source",
            "source_provenance",
            "year",
            "month",
            "basin",
        ]
    )
    present = [column for column in ordered if column in manifest.columns]
    remainder = [column for column in manifest.columns if column not in present]
    manifest = manifest[present + remainder].sort_values(["storm_id", "timestamp"]).reset_index(drop=True)
    return manifest, coverage


def build_manifest(
    digital_typhoon_df: pd.DataFrame,
    ibtracs_df: pd.DataFrame,
    agencies: tuple[str, ...] = AGENCIES,
    basin: str = "WP",
    year_min: int = 1988,
    year_max: int = 2023,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy direct merge helper kept for compatibility with older workflows."""

    dt_df = digital_typhoon_df.copy()
    ib_df = ibtracs_df.copy()
    dt_df = dt_df[dt_df["basin"].fillna("").astype(str).str.upper().eq(basin)]
    ib_df = ib_df[ib_df["basin"].fillna("").astype(str).str.upper().eq(basin)]
    dt_df = dt_df[(dt_df["timestamp"].dt.year >= year_min) & (dt_df["timestamp"].dt.year <= year_max)]
    ib_df = ib_df[(ib_df["timestamp"].dt.year >= year_min) & (ib_df["timestamp"].dt.year <= year_max)]
    manifest = dt_df.merge(
        ib_df.drop(columns=["storm_name", "basin", "lat", "lon"], errors="ignore"),
        how="left",
        on=["storm_id", "timestamp"],
        suffixes=("", "_ib"),
    )
    manifest["source_match_method"] = np.where(manifest["pressure_jma"].notna(), "storm_id", pd.NA)
    manifest["year"] = manifest["timestamp"].dt.year
    manifest["month"] = manifest["timestamp"].dt.month
    if "storm_name_ib" in manifest.columns:
        manifest["storm_name"] = manifest["storm_name"].fillna(manifest["storm_name_ib"])
    if "lat_ib" in manifest.columns:
        manifest["lat"] = manifest["lat"].fillna(manifest["lat_ib"])
    if "lon_ib" in manifest.columns:
        manifest["lon"] = manifest["lon"].fillna(manifest["lon_ib"])
    manifest, coverage = _derive_consensus_columns(manifest, agencies)
    for agency in agencies:
        for column in [f"pressure_{agency}", f"wind_{agency}", f"wind_avg_period_{agency}"]:
            if column not in manifest.columns:
                manifest[column] = pd.NA
    return manifest.sort_values(["storm_id", "timestamp"]).reset_index(drop=True), coverage
