# TyphoonUQ-Bench

**A reproducible multi-agency uncertainty benchmark for typhoon intensity estimation from satellite imagery (Western North Pacific, 1988–2023).**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![Data DOI](https://img.shields.io/badge/data-Zenodo-blueviolet.svg)](https://doi.org/10.5281/zenodo.20482461)

TyphoonUQ-Bench links Digital Typhoon satellite imagery, JMA RSMC best-track records, and the IBTrACS multi-agency archive into a single aligned manifest, defines three uncertainty-quantification tracks (`analysis-0h`, `forecast-6h`, `forecast-12h`) on a fixed chronological year split, and ships three baseline families (five concrete configurations) together with an evaluation suite that scores point accuracy, interval calibration, and range-aware metrics against the inter-agency pressure spread.

---

## Table of contents

1. [Why this benchmark](#1-why-this-benchmark)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Getting the data](#4-getting-the-data)
5. [End-to-end reproduction](#5-end-to-end-reproduction)
6. [Tracks, splits, and the evaluation suite](#6-tracks-splits-and-the-evaluation-suite)
7. [Built-in baselines](#7-built-in-baselines)
8. [Plugging in an external UQ model](#8-plugging-in-an-external-uq-model)
9. [Reproducing benchmark figures](#9-reproducing-benchmark-figures)
10. [Data citation and licensing](#10-data-citation-and-licensing)

---

## 1. Why this benchmark

Operational tropical cyclone (TC) intensity records disagree across regional agencies — JMA, JTWC, CMA, and HKO frequently report different central pressures and maximum sustained winds for the same storm at the same time. Deep-learning intensity studies typically train and evaluate against a single agency, so cross-paper accuracy and uncertainty comparisons remain unreliable.

TyphoonUQ-Bench targets that gap by:

- aligning Digital Typhoon imagery, JMA best-track records, and IBTrACS multi-agency intensity records through a JMA-anchored matching pipeline;
- enforcing a single chronological year split (train 1988–2015 / val 2016–2019 / test 2020–2023) with no storm leakage across folds;
- exposing the inter-agency pressure spread as a structured uncertainty signal that range-aware metrics can probe directly;
- shipping three baseline families (five concrete configurations) and an evaluation entry point so that any external UQ model meeting the benchmark's interface can be scored against identical data and metrics with a single command.

---

## 2. Repository layout

The numbered top-level folders follow the expected reproduction order.

```
TyphoonUQ-Bench/
├── README.md                              # this file
├── LICENSE                                # MIT
├── pyproject.toml                         # package metadata + console scripts
├── .gitignore
│
├── 01_data_preparation/
│   └── 01_build_digital_typhoon_index.py  # storm & frame index from raw archive
│
├── 02_benchmark_construction/
│   ├── 01_build_aligned_manifest.py       # DT ↔ JMA ↔ IBTrACS alignment
│   ├── 02_make_forward_splits.py          # chronological year split
│   └── 03_build_task_samples.py           # per-task sample tables (preferred external interface)
│
├── 03_model_training/
│   ├── 01_train_metadata_baseline.py      # XGBoost / GradientBoosting
│   └── 02_train_torch_baseline.py         # image-only / multimodal, tiny / resnet18
│
├── 04_evaluation/
│   ├── 01_evaluate_predictions.py         # score a built-in baseline's predictions.parquet
│   ├── 02_evaluate_external_predictions.py # score an external (sample_id, prediction) table
│   └── 03_run_case_study.py               # per-storm case study (Haiyan 2013, …)
│
├── 05_reproduction/
│   └── ascend/                            # Ascend NPU driver scripts
│       ├── env_check.sh
│       ├── run_full_pipeline.sh
│       ├── run_torch_ddp.sh
│       ├── run_torch_independent_pool.sh
│       └── wait_for_free_npu_and_run_torch.sh
│
├── 06_paper_figures/
│   ├── 01_make_paper_figures.py           # unified figure generator
│   └── source_data/                       # 8 CSVs that drive every figure
│
├── configs/
│   ├── default.yaml                       # project-wide defaults
│   ├── source_columns.yaml                # IBTrACS / DT column maps
│   └── jma_best_track_notes.md
│
├── data/                                  # placeholder structure (gitignored)
│   ├── README.md
│   ├── raw/{digital_typhoon, jma, ibtracs}/README.md
│   └── processed/{digital_typhoon, benchmark}/README.md
│
├── outputs/                               # placeholder (gitignored)
│   └── README.md
│
├── docs/                                  # extended documentation
│   ├── 03_REPRODUCE.md                    # exact commands for v1 results
│   ├── 04_LICENSES_AND_ATTRIBUTION.md
│   └── 06_ASCEND_REMOTE.md                # optional NPU workflow
│
├── examples/
│   ├── 01_custom_model_template_sklearn.py  # 30-line scikit-learn template
│   ├── 02_custom_model_template_torch.py    # PyTorch DataLoader template
│   └── 03_evaluate_submission.py            # schema validator
│
├── external_models/
│   └── tc_qformer/                        # worked adapter: Guo et al. 2026
│       ├── README.md
│       ├── model.py
│       ├── train_tc_qformer.py
│       ├── aggregate_metrics.py
│       └── run_tc_qformer_npu.sh
│
└── src/typhoonuq/                         # installable Python package
    ├── __init__.py                        # public API
    ├── constants.py, schema.py, io.py, utils.py
    ├── adapters/{digital_typhoon,jma_best_track}.py
    ├── alignment.py                       # DT ↔ JMA ↔ IBTrACS matching
    ├── features.py                        # 8-feature tabular vector × 6 frames
    ├── datasets.py                        # TyphoonClipDataset
    ├── splits.py                          # chronological & LOAO splits
    ├── conformal.py                       # residual-quantile calibration
    ├── metrics.py                         # 9 metrics: point / interval / range-aware
    ├── baselines/{metadata,torch_models}.py
    └── cli.py                             # console entry-point implementations
```

Folders that are **not** committed (regenerated locally or fetched from Zenodo): `data/raw/` (Digital Typhoon archive, IBTrACS CSV, JMA best-track text), `data/processed/` (derived manifests and match tables), and `outputs/` (training runs, prediction parquets, metrics JSON).

The companion Zenodo bundle ships the contents of `data/processed/` and `outputs/splits/` so users can skip the build phase entirely.

---

## 3. Installation

Python ≥ 3.10 is required.

```bash
git clone https://github.com/janesolari82-dotcom/TyphoonUQ-Bench.git
cd TyphoonUQ-Bench

# Editable install
python -m pip install -e .

# ML extras (torch, torchvision, xgboost) for the deep-learning baselines
python -m pip install -e ".[ml]"

# Figure-rendering extras (matplotlib, scipy, seaborn)
python -m pip install -e ".[figures]"

# Optional developer tools
python -m pip install -e ".[dev]"
```

After installation, these console entry points are also available:

| Command | Equivalent script |
|---------|-------------------|
| `typhoonuq-dt-build-index` | `01_data_preparation/01_build_digital_typhoon_index.py` |
| `typhoonuq-build-manifest` | `02_benchmark_construction/01_build_aligned_manifest.py` |
| `typhoonuq-make-splits`    | `02_benchmark_construction/02_make_forward_splits.py`   |
| `typhoonuq-train-metadata` | `03_model_training/01_train_metadata_baseline.py`       |
| `typhoonuq-train-torch`    | `03_model_training/02_train_torch_baseline.py`          |

---

## 4. Getting the data

### 4.1 Raw archives (download yourself)

| Archive | Source | Notes |
|---------|--------|-------|
| Digital Typhoon | <http://agora.ex.nii.ac.jp/digital-typhoon/> | DIAS distribution; CC-BY-4.0. |
| JMA RSMC best-track | <https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/besttrack.html> | `bst_all.txt`; Japan Public Data License 1.0 / CC-BY-4.0-compatible. |
| IBTrACS v04r01 (WP) | <https://www.ncei.noaa.gov/products/international-best-track-archive> | `ibtracs.WP.list.v04r01.csv`; NOAA/NCEI public/open data. |

Place the downloads exactly as:

```
data/raw/
├── digital_typhoon/archive/{metadata.json, metadata/, image/, image_png/, wnp/}
├── jma/bst_all.txt
└── ibtracs/ibtracs.WP.list.v04r01.csv
```

The per-folder `README.md` files inside `data/` reproduce this layout for new users.

### 4.2 Derived artefacts (Zenodo mirror)

If you only want to consume the benchmark without re-running the build pipeline, fetch the companion Zenodo bundle (`Zenodo/` mirror in this release; DOI in [§10](#10-data-citation-and-licensing)). The derived parquets are binary-identical to what `02_benchmark_construction/01_build_aligned_manifest.py` and `02_benchmark_construction/02_make_forward_splits.py` produce locally:

```
benchmark/benchmark_manifest.parquet                  8.2 MB
benchmark/{storm,dt_jma,jma_ibtracs}_match_table.parquet
benchmark/{coverage_report,manual_review}.{parquet,csv}
splits/{forward_main,leave_one_agency_out}.parquet
sample_cache/{analysis-0h,forecast-6h,forecast-12h}_png_h6_samples.parquet
```

Drop them under `data/processed/benchmark/`, `outputs/splits/`, and `outputs/splits/sample_cache/` to skip steps 1–4 of the rebuild.

---

## 5. End-to-end reproduction

The pipeline runs in five stages. Each numbered folder is a stage; each numbered script inside a folder is a step within that stage.

```bash
# Stage 1 — Data preparation
python 01_data_preparation/01_build_digital_typhoon_index.py \
    --archive-root data/raw/digital_typhoon/archive \
    --output-dir   data/processed/digital_typhoon \
    --year-min 1988 --year-max 2023

# Stage 2 — Benchmark construction
python 02_benchmark_construction/01_build_aligned_manifest.py \
    --storm-index   data/processed/digital_typhoon/storm_index.parquet \
    --frame-index   data/processed/digital_typhoon/frame_index.parquet \
    --jma-best-track data/raw/jma/bst_all.txt \
    --ibtracs-csv   data/raw/ibtracs/ibtracs.WP.list.v04r01.csv \
    --source-columns configs/source_columns.yaml \
    --output-dir    data/processed/benchmark

python 02_benchmark_construction/02_make_forward_splits.py \
    --manifest   data/processed/benchmark/benchmark_manifest.parquet \
    --output-dir outputs/splits

python 02_benchmark_construction/03_build_task_samples.py \
    --manifest   data/processed/benchmark/benchmark_manifest.parquet \
    --split-file outputs/splits/forward_main.parquet \
    --image-backend png \
    --output-dir outputs/splits/sample_cache

# Stage 3 — Train a reference baseline
python 03_model_training/02_train_torch_baseline.py \
    --manifest    data/processed/benchmark/benchmark_manifest.parquet \
    --split-file  outputs/splits/forward_main.parquet \
    --data-root   data/raw/digital_typhoon/archive \
    --task        analysis-0h \
    --mode        image-only \
    --backbone    resnet18 \
    --image-backend png \
    --output-dir  outputs/torch_image_resnet18/analysis-0h

# Stage 4 — Score predictions
python 04_evaluation/01_evaluate_predictions.py \
    --predictions outputs/torch_image_resnet18/analysis-0h/predictions.parquet \
    --output-json outputs/torch_image_resnet18/analysis-0h/eval_metrics.json
```

The full 5×3 baseline grid and the TC-QFormer adapter walk-through live in [`docs/03_REPRODUCE.md`](docs/03_REPRODUCE.md).

---

## 6. Tracks, splits, and the evaluation suite

**Tracks.** The same manifest serves three tracks that differ only in their prediction horizon:

| Track          | Horizon | Target |
|----------------|---------|--------|
| `analysis-0h`  | 0 h     | Consensus central pressure at `t` (hPa). |
| `forecast-6h`  | +6 h    | Consensus central pressure at `t + 6 h`. |
| `forecast-12h` | +12 h   | Consensus central pressure at `t + 12 h`. |

Each sample has 6 consecutive 1-hourly satellite frames (`t-5 … t-0`) plus the matching 8-feature tabular vector.

**Splits.** Chronological year split with no storm leakage:

- train: 1988–2015 — 706 storms / 143 221 frames
- val:   2016–2019 — 110 storms / 20 585 frames
- test:  2020–2023 — 87 storms  / 16 557 frames

**Metrics.** Three layers, nine numbers in total:

- *Point*: `mae`, `rmse`.
- *Interval*: `interval_coverage`, `interval_width`, `calibration_error`.
- *Range-aware*: `in_range_rate`, `distance_to_range`, `range_coverage_at_90`, `dispersion_interval_corr`.

Range-aware metrics compare each model interval against the per-sample inter-agency pressure range. Metric definitions are given in §8 below.

---

## 7. Built-in baselines

TyphoonUQ-Bench ships **three baseline families** — `metadata-only`, `image-only`, and `multimodal` — exposed through the `03_model_training/02_train_torch_baseline.py` CLI as `--mode {image-only, multimodal}` together with `--backbone {tiny, resnet18}`. The five concrete configurations used in the v1 reference run are:

| Configuration         | Family       | Backbone                        | Modality        | Notes |
|-----------------------|--------------|---------------------------------|-----------------|-------|
| `metadata`            | metadata-only| XGBoost (sklearn fallback)      | tabular only    | 8 features × 6 time steps |
| `image_tiny`          | image-only   | TinyFrameEncoder (3-layer CNN)  | image only      | GRU / TCN / mean aggregator |
| `image_resnet18`      | image-only   | ResNet18 (single-channel)       | image only      | GRU / TCN / mean aggregator |
| `multimodal_tiny`     | multimodal   | TinyFrameEncoder + MLP          | image + tabular | concat before regression head |
| `multimodal_resnet18` | multimodal   | ResNet18 + MLP                  | image + tabular | concat before regression head |

All deep-learning baselines train with Gaussian NLL (quantile loss also available), use batch size 4, learning rate 3 × 10⁻⁴, ReduceLROnPlateau, and early stopping with patience 2. They emit a final 80% / 90% prediction interval whose bounds are the *wider* of the model's native bound and a post-hoc residual-conformal bound on the validation split.

---

## 8. Plugging in an external UQ model

TyphoonUQ-Bench supports two integration paths. Pick whichever matches the structure of your model.

### 8.1 Lightweight path — sample-table interface

Recommended for models that consume tabular features only, or for quick sanity checks before a heavier integration.

```bash
# 1. Once: materialise per-task sample tables to parquet.
python 02_benchmark_construction/03_build_task_samples.py \
    --manifest   data/processed/benchmark/benchmark_manifest.parquet \
    --split-file outputs/splits/forward_main.parquet \
    --image-backend png \
    --output-dir outputs/splits/sample_cache

# 2. Train your model on the table, save predictions as
#    (sample_id, prediction[, lower, upper, lower_90, upper_90]) parquet/CSV.
#    See examples/01_custom_model_template_sklearn.py for a 30-line example.

# 3. Score it.
python 04_evaluation/02_evaluate_external_predictions.py \
    --samples     outputs/splits/sample_cache/forecast-6h_png_h6_samples.parquet \
    --predictions outputs/custom_model_predictions.parquet \
    --split       test \
    --output-json outputs/custom_model_metrics.json
```

### 8.2 Full path — image + tabular DataLoader

Recommended for image-based UQ models that need access to the original PNG/H5 frames. `examples/02_custom_model_template_torch.py` provides a PyTorch DataLoader skeleton; replace the body of `predict_one_batch` with your own model. `external_models/tc_qformer/` is a complete worked adapter that wraps TC-QFormer (Guo et al. 2026) and reuses the entire benchmark training stack.

In both paths, the predictions file must contain at minimum `sample_id` and `prediction`. Optional columns (`lower`, `upper`, `lower_90`, `upper_90`) unlock the interval and range-aware metrics. Before scoring, run the schema check:

```bash
python examples/03_evaluate_submission.py \
    --predictions outputs/custom_model_predictions.parquet
```

### 8.3 Evaluation metrics

`src/typhoonuq/metrics.py::evaluate_predictions` returns one dict per call. Pure NumPy/Pandas implementation, negligible overhead on millions of rows.

**Point metrics**

| Key | Formula |
|-----|---------|
| `mae`  | mean(\|target − prediction\|) |
| `rmse` | sqrt(mean((target − prediction)²)) |

**Interval metrics** — computed at the configured `target_coverage` (default 0.80; pass `--target-coverage` to override).

| Key | Description |
|-----|-------------|
| `interval_coverage` | Fraction of test rows with `lower ≤ target ≤ upper`. Ideal value = `target_coverage`. |
| `interval_width`    | Mean Prediction Interval Width (MPIW) in hPa. Lower is sharper. |
| `calibration_error` | Absolute discrepancy between empirical and nominal coverage at the configured `target_coverage` level: \|empirical_coverage − target_coverage\|. |

**Range-aware metrics** — require `target_lower` / `target_upper`; `dispersion_interval_corr` additionally requires a non-NaN `pressure_range` column.

| Key | Description |
|-----|-------------|
| `in_range_rate`         | Fraction of `prediction` values that fall inside `[target_lower, target_upper]`. |
| `distance_to_range`     | Mean of `max(0, target_lower − prediction) + max(0, prediction − target_upper)`. Zero for in-range predictions. |
| `range_coverage_at_90`  | Fraction of 90%-nominal intervals that fully contain `[target_lower, target_upper]`. |
| `dispersion_interval_corr` | Pearson correlation between `(upper − lower)` and `(target_upper − target_lower)` on samples where `pressure_range` is non-NaN and non-zero. Positive values indicate the model widens its interval when the agencies disagree. |

A typical output (the v1 reference run of `image_resnet18` on `forecast-12h`, scored on the test split):

```json
{
  "mae": 7.7680,
  "rmse": 10.7397,
  "interval_coverage": 0.9650,
  "interval_width": 51.4813,
  "calibration_error": 0.1650,
  "in_range_rate": 0.1522,
  "distance_to_range": 6.0920,
  "range_coverage_at_90": 0.8635,
  "dispersion_interval_corr": 0.1002
}
```

---

## 9. Reproducing benchmark figures

The benchmark figures are driven by the eight CSVs under `06_paper_figures/source_data/`. The figure script reads these CSVs (not training outputs), so figures can be regenerated even if model checkpoints are not available.

```bash
python 06_paper_figures/01_make_paper_figures.py
```

To regenerate the source CSVs themselves, re-train the baselines and use the helpers documented inside `01_make_paper_figures.py`.

---

## 10. Data citation and licensing

### 10.1 Citing the dataset

```bibtex
@dataset{typhoonuq_data_2026,
  title   = {{TyphoonUQ-Bench} v1: aligned manifest, splits, and sample caches},
  author  = {Gao, Yang and Su, Junchao and Zhou, Jingzhi and Meng, Fan},
  year    = {2026},
  doi     = {10.5281/zenodo.20482461},
  url     = {https://doi.org/10.5281/zenodo.20482461},
}
```

### 10.2 Source-data attribution

TyphoonUQ-Bench is a *derived* dataset built from three upstream archives. The benchmark itself is released under MIT, but the upstream sources retain their own terms. See [`docs/04_LICENSES_AND_ATTRIBUTION.md`](docs/04_LICENSES_AND_ATTRIBUTION.md) for the per-source breakdown.

### 10.3 Contact

Issues and pull requests are welcome via GitHub. For data-alignment questions or to propose a new external adapter, please open a discussion thread referencing the relevant prior work.
