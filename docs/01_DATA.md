# Data sources, alignment, and manifest schema

This document describes the three upstream archives that TyphoonUQ-Bench consumes, the JMA-anchored alignment pipeline that merges them, and the schema of the benchmark manifest that downstream baselines read.

---

## 1. Upstream archives

### 1.1 Digital Typhoon

- Source: <http://agora.ex.nii.ac.jp/digital-typhoon/>
- Distribution: National Institute of Informatics (NII) DIAS
- Coverage: 1978-present, Western North Pacific basin, geostationary IR
- License: CC-BY-4.0

We use:

```
data/raw/digital_typhoon/archive/
├── image_png/image_png/<storm_id>/   # 5-km grid, single-channel IR PNG
└── wnp/h5/                            # optional HDF5 backend with brightness temperatures
```

`<storm_id>` is a 6-character string of the form `YYYYNN` where `YYYY` is the season and `NN` is the per-year storm index. Each storm directory contains one PNG (and optionally one HDF5) per typhoon-centred frame, at 1-hourly cadence.

### 1.2 JMA RSMC best-track

- Source: <https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/besttrack.html>
- Format: plain text (`bst_all.txt`), fixed-width
- Coverage: 1951-2026 (annually updated by RSMC Tokyo)
- License: CC-BY-4.0-compatible (Japan Public Data License 1.0)

Parsed by `src/typhoonuq/adapters/jma_best_track.py`. The parser tolerates RSMC formatting quirks (mid-line header records, missing fields), produces one row per `(storm_id, timestamp)` pair, and emits both the cross-year unique `jma_storm_id` and the season-local `jma_tc_id` for traceability.

### 1.3 IBTrACS v04r01

- Source: <https://www.ncei.noaa.gov/products/international-best-track-archive>
- File: `ibtracs.WP.list.v04r01.csv` (Western Pacific subset)
- Provides: multi-agency intensity records (JMA, JTWC, CMA, HKO, …)
- License: NOAA/NCEI public/open data

Loaded via `src/typhoonuq/alignment.py::standardize_ibtracs`, which renames agency-specific columns through `configs/source_columns.yaml`.

---

## 2. JMA-anchored alignment pipeline

```
Digital Typhoon storms
        │
        │  fuzzy match by (season, name, track similarity)
        ▼
JMA RSMC best-track storms  ◄──── anchor
        │
        │  fuzzy match by (season, name, track similarity)
        ▼
IBTrACS multi-agency rows
        │
        ▼
benchmark_manifest.parquet   (one row per (storm, timestamp))
```

The pipeline is implemented in `src/typhoonuq/alignment.py`:

1. `build_dt_jma_match_table` — Digital Typhoon storms → JMA storms.
2. `build_jma_ibtracs_match_table` — JMA storms → IBTrACS storms.
3. `build_anchored_storm_match_table` — composes the two match tables into a single storm-level lookup.
4. `build_benchmark_manifest` — joins frame-level Digital Typhoon records with JMA/IBTrACS intensities, computes the consensus target, and emits `benchmark_manifest.parquet`.

### 2.1 Consensus pressure target

The consensus central pressure is the per-timestamp median across agencies whose coverage in the Western Pacific exceeds a configurable threshold (`agency_coverage_threshold = 0.25` in `configs/default.yaml`). On the v1 release this filter selects JMA and CMA; JTWC and HKO are retained as reference columns but do not contribute to the consensus.

When only one of the two qualifying agencies reports for a sample, `pressure_range = 0` and `agency_count = 1`. When both report, `agency_count = 2` and `pressure_range ≥ 0` (positive when the two agencies report different values; zero when both happen to agree on the same pressure).

---

## 3. Manifest schema

`benchmark_manifest.parquet` is the canonical artefact. One row per `(storm_id, timestamp)` pair.

### 3.1 Identification

| Column | Type | Description |
|--------|------|-------------|
| `storm_id`    | string  | Digital Typhoon storm key (`YYYYNN`).        |
| `storm_name`  | string  | Storm name (may be `NONAME`).                |
| `timestamp`   | datetime[ns, UTC] | Frame timestamp (1-hourly). |
| `year`, `month` | int   | Convenience date parts.                      |

### 3.2 Image references

| Column      | Type   | Description |
|-------------|--------|-------------|
| `image_ref` | string | Backend-agnostic identifier (PNG path stem). |
| `png_ref`   | string | Path under `data/raw/digital_typhoon/archive/image_png/image_png/`. |
| `h5_ref`    | string | Path under `data/raw/digital_typhoon/archive/wnp/h5/` (nullable). |
| `quality_flag` | int | Digital Typhoon per-frame quality bit.    |
| `lat`, `lon`   | float | Typhoon centre at this timestamp.        |

### 3.3 Per-agency intensity

For each agency in `(jma, jtwc, cma, hko)`:

| Column                    | Type  | Description                          |
|---------------------------|-------|--------------------------------------|
| `pressure_<agency>`       | float | Reported central pressure (hPa).     |
| `wind_<agency>`           | float | Reported MSW (knots).                |
| `wind_avg_period_<agency>`| string | `10min`, `1min`, or empty when unspecified. |

### 3.4 Consensus columns

| Column                          | Type  | Description |
|---------------------------------|-------|-------------|
| `pressure_consensus_median`     | float | **Target** of the benchmark (hPa). |
| `pressure_min`, `pressure_max`  | float | Min / max across qualifying agencies (hPa). |
| `pressure_range`                | float | `pressure_max - pressure_min` (hPa). |
| `pressure_std`                  | float | Std-dev across qualifying agencies (hPa). |
| `agency_count`                  | int   | Number of agencies contributing to consensus (max = 2 on v1). |
| `metric_agencies`               | string| Comma-separated list of selected agencies. |

### 3.5 Schema validation

```python
from typhoonuq.io import load_manifest
from typhoonuq.schema import validate_manifest

df = load_manifest("data/processed/benchmark/benchmark_manifest.parquet")
validate_manifest(df)  # raises on any missing column or wrong dtype
```

---

## 4. Split file schema

`outputs/splits/forward_main.parquet` adds a `split` column to a subset of the manifest, restricted to storms that have enough usable frames for the 6-hour history window. Columns:

| Column     | Type   | Description |
|------------|--------|-------------|
| `storm_id` | string | Same as the manifest. |
| `timestamp`| datetime[ns, UTC] | Same as the manifest. |
| `year`     | int    | Derived from `timestamp` for split-band lookup. |
| `split`    | string | One of `train` / `val` / `test`. |

The split file is task-agnostic: the same `(storm_id, timestamp)` row is used by every track, with the prediction horizon supplied at runtime via the `--task` flag of each training script. Generated by `02_benchmark_construction/02_make_forward_splits.py` and consumed by every training script via `TyphoonClipDataset(split_df=...)`.

---

## 5. Sample cache

For the PNG backend the splits script also writes a sample-level cache (`outputs/splits/sample_cache/<task>_png_h6_samples.parquet`). Each row fully expands the per-sample tabular vector (8 features × 6 time steps = 48 numeric columns named `<feature>_t-<n>`) plus the target and the agency metadata. Loading the cache instead of recomputing the tabular vectors reduces training-startup latency by ~10×.

Some rows reference image paths of the form `image_png/.../nan.png` in their `png_refs` field — these are placeholders for missing frames inside the 6-hour history window. `TyphoonClipDataset` treats them as missing inputs and substitutes a zero image; they do **not** indicate corrupted data.
