"""Adapter for JMA/RSMC best track text files."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from typhoonuq.utils import ensure_utc_timestamp, normalize_name

_HEADER_RE = re.compile(
    r"^66666\s+"
    r"(?P<intl_id>\d{4})\s+"
    r"(?P<count>\d{1,3})\s+"
    r"(?:(?P<tc_id>\d{4})\s+)?"
    r"(?P<intl_id_rep>\d{4})\s+"
    r"(?P<last_flag>\d)\s+"
    r"(?P<diff>\d)\s+"
    r"(?P<name>.{0,20}?)\s+"
    r"(?P<revision>\d{8})\s*$"
)

_DATA_RE = re.compile(
    r"^(?P<time>\d{8})\s+"
    r"(?P<indicator>\d{3})\s+"
    r"(?P<grade>\d)\s+"
    r"(?P<lat>\d{3})\s+"
    r"(?P<lon>\d{4})\s+"
    r"(?P<pressure>[0-9 ]{4})\s+"
    r"(?P<wind>[0-9 ]{3})\s+"
    r"(?P<dir50>[0-9 ])(?P<long50>[0-9 ]{4})\s"
    r"(?P<short50>[0-9 ]{4})\s"
    r"(?P<dir30>[0-9 ])(?P<long30>[0-9 ]{4})\s"
    r"(?P<short30>[0-9 ]{4})\s*"
    r"(?P<landfall>#?)\s*$"
)


def _parse_int(text: str) -> int | None:
    text = text.strip()
    return int(text) if text else None


def _yy_to_year(value: str) -> int:
    yy = int(value)
    return 1900 + yy if yy >= 51 else 2000 + yy


def _parse_timestamp(yyMMddHH: str) -> pd.Timestamp:
    year = _yy_to_year(yyMMddHH[:2])
    month = yyMMddHH[2:4]
    day = yyMMddHH[4:6]
    hour = yyMMddHH[6:8]
    timestamp = pd.Timestamp(f"{year:04d}-{month}-{day} {hour}:00:00", tz="UTC")
    return timestamp.tz_convert(None)


def _parse_header(line: str, line_no: int) -> dict[str, object]:
    header_match = _HEADER_RE.match(line)
    if header_match is None:
        raise ValueError(f"Unrecognized JMA header at line {line_no}: {line!r}")
    intl_id = header_match.group("intl_id")
    intl_id_rep = header_match.group("intl_id_rep")
    if intl_id != intl_id_rep:
        raise ValueError(
            f"JMA header id mismatch at line {line_no}: intl_id={intl_id!r}, intl_id_rep={intl_id_rep!r}"
        )
    return {
        "jma_storm_id": intl_id,
        "jma_tc_id": header_match.group("tc_id") or pd.NA,
        "storm_name": header_match.group("name").strip() or pd.NA,
        "revision_date": header_match.group("revision"),
        "declared_count": int(header_match.group("count")),
        "season": _yy_to_year(intl_id[:2]),
    }


def _append_block_rows(
    header: dict[str, object],
    block_rows: list[tuple[int, re.Match[str]]],
    rows: list[dict[str, object]],
) -> None:
    declared = int(header["declared_count"])
    if declared != len(block_rows):
        raise ValueError(
            "JMA storm block row count mismatch for "
            f"{header['jma_storm_id']}: declared {declared}, parsed {len(block_rows)}"
        )
    for line_no, data_match in block_rows:
        timestamp = _parse_timestamp(data_match.group("time"))
        rows.append(
            {
                "jma_storm_id": header["jma_storm_id"],
                "jma_tc_id": header["jma_tc_id"],
                "storm_name": header["storm_name"],
                "storm_name_key": normalize_name(header["storm_name"]),
                # Keep storm season stable across year boundaries.
                "season": int(header["season"]),
                "timestamp": timestamp,
                "lat": _parse_int(data_match.group("lat")) / 10.0 if _parse_int(data_match.group("lat")) is not None else pd.NA,
                "lon": _parse_int(data_match.group("lon")) / 10.0 if _parse_int(data_match.group("lon")) is not None else pd.NA,
                "pressure_jma_native": _parse_int(data_match.group("pressure")),
                "wind_jma_native": _parse_int(data_match.group("wind")),
                "wind_avg_period_jma_native": "10min",
                "jma_grade": _parse_int(data_match.group("grade")),
                "indicator": _parse_int(data_match.group("indicator")),
                "r50_dir": _parse_int(data_match.group("dir50")),
                "r50_long": _parse_int(data_match.group("long50")),
                "r50_short": _parse_int(data_match.group("short50")),
                "r30_dir": _parse_int(data_match.group("dir30")),
                "r30_long": _parse_int(data_match.group("long30")),
                "r30_short": _parse_int(data_match.group("short30")),
                "landfall_flag": True if data_match.group("landfall") == "#" else False,
                "revision_date": header["revision_date"],
                "basin": "WP",
            }
        )


def load_jma_best_track(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    rows: list[dict[str, object]] = []
    current_header: dict[str, object] | None = None
    current_block: list[tuple[int, re.Match[str]]] = []
    seen_storm_ids: set[str] = set()

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("66666"):
            if current_header is not None:
                _append_block_rows(current_header, current_block, rows)
            current_header = _parse_header(line, line_no)
            storm_id = str(current_header["jma_storm_id"])
            if storm_id in seen_storm_ids:
                raise ValueError(f"Duplicate JMA storm id encountered at line {line_no}: {storm_id}")
            seen_storm_ids.add(storm_id)
            current_block = []
            continue
        data_match = _DATA_RE.match(line)
        if current_header is None:
            raise ValueError(f"Encountered JMA data before any header at line {line_no}: {line!r}")
        if data_match is None:
            raise ValueError(f"Unrecognized JMA data line at line {line_no}: {line!r}")
        current_block.append((line_no, data_match))

    if current_header is not None:
        _append_block_rows(current_header, current_block, rows)

    if not rows:
        return pd.DataFrame(
            columns=[
                "jma_storm_id",
                "jma_tc_id",
                "storm_name",
                "storm_name_key",
                "season",
                "timestamp",
                "lat",
                "lon",
                "pressure_jma_native",
                "wind_jma_native",
                "wind_avg_period_jma_native",
                "jma_grade",
                "indicator",
                "r50_dir",
                "r50_long",
                "r50_short",
                "r30_dir",
                "r30_long",
                "r30_short",
                "landfall_flag",
                "revision_date",
                "basin",
            ]
        )

    df = pd.DataFrame(rows)
    df["timestamp"] = ensure_utc_timestamp(df["timestamp"])
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["pressure_jma_native"] = pd.to_numeric(df["pressure_jma_native"], errors="coerce")
    df["wind_jma_native"] = pd.to_numeric(df["wind_jma_native"], errors="coerce")
    return df.sort_values(["season", "jma_storm_id", "timestamp"]).reset_index(drop=True)
