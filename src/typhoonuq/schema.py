"""Manifest schema validation."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .constants import AGENCIES, CONSENSUS_COLUMNS, CORE_MANIFEST_COLUMNS


@dataclass
class ValidationResult:
    missing_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_columns


def required_manifest_columns(agencies: tuple[str, ...] = AGENCIES) -> list[str]:
    columns = list(CORE_MANIFEST_COLUMNS) + list(CONSENSUS_COLUMNS)
    for agency in agencies:
        columns.extend(
            [
                f"pressure_{agency}",
                f"wind_{agency}",
                f"wind_avg_period_{agency}",
            ]
        )
    return columns


def validate_manifest(df: pd.DataFrame, strict: bool = True) -> ValidationResult:
    missing = [column for column in required_manifest_columns() if column not in df.columns]
    if "image_ref" in df.columns:
        if "png_ref" in missing:
            missing.remove("png_ref")
        if "h5_ref" in missing:
            missing.remove("h5_ref")
    if "png_ref" in missing and "h5_ref" in df.columns:
        missing.remove("png_ref")
    if "h5_ref" in missing and "png_ref" in df.columns:
        missing.remove("h5_ref")
    warnings: list[str] = []
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        warnings.append("`timestamp` is not a datetime column.")
    if "year" in df.columns and "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        mismatched = (df["year"].fillna(-1).astype(int) != df["timestamp"].dt.year.fillna(-1).astype(int)).sum()
        if mismatched:
            warnings.append(f"`year` disagrees with `timestamp` on {mismatched} rows.")
    result = ValidationResult(missing_columns=missing, warnings=warnings)
    if strict and missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    return result
