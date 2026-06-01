# Reproducing the v1 benchmark

This is the exact command sequence we ran to produce the numbers in the
TyphoonUQ-Bench Letter (FCS Code & Data, 2026). Everything in this file
should run on a single workstation with an Ascend NPU, an NVIDIA GPU, or
just CPU (slow but functional).

---

## 0. Environment

```bash
# Python 3.10 or newer.
python --version

# Install the benchmark package together with PyTorch, XGBoost, matplotlib.
python -m pip install -e ".[ml,figures]"
```

Approximate disk budget (after stage 2):

| Path                                | Size  |
|-------------------------------------|-------|
| `data/raw/digital_typhoon/archive/` | ~120 GB (PNG + H5) |
| `data/raw/jma/bst_all.txt`          | ~5 MB |
| `data/raw/ibtracs/...csv`           | ~14 MB |
| `data/processed/benchmark/`         | ~9 MB |
| `outputs/splits/`                   | ~30 MB |
| `outputs/<baseline>/<task>/`        | ~1 MB per task |

Skipping data construction? Drop the Zenodo bundle into
`data/processed/benchmark/`, `outputs/splits/`, and
`outputs/splits/sample_cache/` and start from stage 3.

---

## Stage 1 — Build Digital Typhoon indexes

```bash
python 01_data_preparation/01_build_digital_typhoon_index.py \
    --archive-root data/raw/digital_typhoon/archive \
    --output-dir   data/processed/digital_typhoon \
    --year-min     1988 \
    --year-max     2023
```

Produces `storm_index.parquet`, `frame_index.parquet`, and a `summary.csv`
showing how many frames were discovered per storm.

---

## Stage 2 — Build the aligned manifest, splits, and sample tables

### 2.1 Aligned manifest

```bash
python 02_benchmark_construction/01_build_aligned_manifest.py \
    --storm-index    data/processed/digital_typhoon/storm_index.parquet \
    --frame-index    data/processed/digital_typhoon/frame_index.parquet \
    --jma-best-track data/raw/jma/bst_all.txt \
    --ibtracs-csv    data/raw/ibtracs/ibtracs.WP.list.v04r01.csv \
    --source-columns configs/source_columns.yaml \
    --basin          WP \
    --year-min       1988 \
    --year-max       2023 \
    --output-dir     data/processed/benchmark
```

Key outputs:

- `benchmark_manifest.parquet` — 180 363 rows on the v1 release.
- `storm_match_table.parquet`, `dt_jma_match_table.parquet`,
  `jma_ibtracs_match_table.parquet` — intermediate match tables.
- `coverage_report.parquet` — per-agency per-year coverage rates used to
  decide which agencies are eligible for the consensus.

### 2.2 Chronological split

```bash
python 02_benchmark_construction/02_make_forward_splits.py \
    --manifest   data/processed/benchmark/benchmark_manifest.parquet \
    --output-dir outputs/splits
```

Outputs:

- `forward_main.parquet` — the canonical split (used everywhere downstream).
- `leave_one_agency_out.parquet` — auxiliary split for sensitivity studies.

### 2.3 Sample tables (used by the lightweight evaluation path)

```bash
python 02_benchmark_construction/03_build_task_samples.py \
    --manifest      data/processed/benchmark/benchmark_manifest.parquet \
    --split-file    outputs/splits/forward_main.parquet \
    --image-backend png \
    --history-hours 6 \
    --output-dir    outputs/splits/sample_cache
```

Outputs one parquet per task:
`outputs/splits/sample_cache/<task>_png_h6_samples.parquet`.

---

## Stage 3 — Train all five built-in baselines for all three tasks

The pattern is identical for every (baseline, task) combination. Below is
the full grid — fifteen runs in total.

### 3.1 Metadata baseline (× 3 tasks)

```bash
for TASK in analysis-0h forecast-6h forecast-12h; do
  python 03_model_training/01_train_metadata_baseline.py \
      --manifest      data/processed/benchmark/benchmark_manifest.parquet \
      --split-file    outputs/splits/forward_main.parquet \
      --task          ${TASK} \
      --image-backend png \
      --output-dir    outputs/metadata/${TASK}
done
```

### 3.2 Image-only baselines (× 2 backbones × 3 tasks)

```bash
for BACKBONE in tiny resnet18; do
  for TASK in analysis-0h forecast-6h forecast-12h; do
    python 03_model_training/02_train_torch_baseline.py \
        --manifest       data/processed/benchmark/benchmark_manifest.parquet \
        --split-file     outputs/splits/forward_main.parquet \
        --data-root      data/raw/digital_typhoon/archive \
        --task           ${TASK} \
        --image-backend  png \
        --mode           image-only \
        --backbone       ${BACKBONE} \
        --probabilistic  gaussian \
        --device         auto \
        --output-dir     outputs/torch_image_${BACKBONE}/${TASK}
  done
done
```

### 3.3 Multimodal baselines (× 2 backbones × 3 tasks)

```bash
for BACKBONE in tiny resnet18; do
  for TASK in analysis-0h forecast-6h forecast-12h; do
    python 03_model_training/02_train_torch_baseline.py \
        --manifest       data/processed/benchmark/benchmark_manifest.parquet \
        --split-file     outputs/splits/forward_main.parquet \
        --data-root      data/raw/digital_typhoon/archive \
        --task           ${TASK} \
        --image-backend  png \
        --mode           multimodal \
        --backbone       ${BACKBONE} \
        --probabilistic  gaussian \
        --device         auto \
        --output-dir     outputs/torch_multimodal_${BACKBONE}/${TASK}
  done
done
```

Each run produces a `metrics.json` (in-process evaluation) and a
`predictions.parquet` (for re-scoring with custom flags).

---

## Stage 4 — Re-score and aggregate

### 4.1 Re-score one prediction file

```bash
python 04_evaluation/01_evaluate_predictions.py \
    --predictions outputs/torch_image_resnet18/analysis-0h/predictions.parquet \
    --target-coverage 0.8 \
    --output-json outputs/torch_image_resnet18/analysis-0h/eval_metrics.json
```

### 4.2 Score an external (sample_id, prediction) file

```bash
python 04_evaluation/02_evaluate_external_predictions.py \
    --samples     outputs/splits/sample_cache/analysis-0h_png_h6_samples.parquet \
    --predictions outputs/custom_model_predictions.parquet \
    --split       test \
    --output-json outputs/custom_model_metrics.json
```

### 4.3 Aggregate the full 5×3 grid

```bash
python -c "
import json, pathlib, pandas as pd
rows = []
for d in sorted(pathlib.Path('outputs').glob('*/*/metrics.json')):
    baseline, task = d.parent.parent.name, d.parent.name
    m = json.loads(d.read_text())['metrics_by_split']['test']
    rows.append({'baseline': baseline, 'task': task, **{k: m[k] for k in
        ('mae','rmse','interval_coverage','interval_width','calibration_error',
         'in_range_rate','distance_to_range','range_coverage_at_90',
         'dispersion_interval_corr')}})
pd.DataFrame(rows).to_csv('outputs/baselines_summary.csv', index=False)
print(open('outputs/baselines_summary.csv').read())
"
```

### 4.4 Storm-level case study

`--predictions` accepts a single `predictions.parquet` file, so run the
script once per task you want to overlay. The example below produces the
manifest-only slice plus one prediction overlay per track:

```bash
# Manifest-only slice (no predictions to overlay)
python 04_evaluation/03_run_case_study.py \
    --manifest    data/processed/benchmark/benchmark_manifest.parquet \
    --storm-id    201330 \
    --output-dir  outputs/case_study/haiyan_2013_manifest

# Overlay each task's predictions
for TASK in analysis-0h forecast-6h forecast-12h; do
  python 04_evaluation/03_run_case_study.py \
      --manifest    data/processed/benchmark/benchmark_manifest.parquet \
      --predictions outputs/torch_image_resnet18/${TASK}/predictions.parquet \
      --storm-id    201330 \
      --output-dir  outputs/case_study/haiyan_2013_image_resnet18_${TASK//-/}
done
```

`storm_id=201330` is Typhoon Haiyan (2013). The script writes per-storm
CSVs and PNGs into the chosen `--output-dir`.

---

## Stage 5 — Reproduce the TC-QFormer external adapter

End-to-end on Ascend NPU:

```bash
PROJECT_ROOT=$(pwd) \
ARCHIVE_ROOT=$(pwd)/data/raw/digital_typhoon/archive \
BATCH_SIZE=8 NUM_WORKERS=0 \
STAGE1_EPOCHS=15 STAGE2_EPOCHS=35 \
bash external_models/tc_qformer/run_tc_qformer_npu.sh
```

CPU smoke test (~5 min):

```bash
python external_models/tc_qformer/train_tc_qformer.py \
    --manifest   data/processed/benchmark/benchmark_manifest.parquet \
    --split-file outputs/splits/forward_main.parquet \
    --data-root  data/raw/digital_typhoon/archive \
    --task       analysis-0h \
    --device     cpu \
    --batch-size 2 \
    --stage1-epochs 1 --stage2-epochs 1 \
    --max-train-batches 5 --max-eval-batches 2 \
    --output-dir outputs/external/tc_qformer/smoke
```

Aggregate across the three tasks:

```bash
python external_models/tc_qformer/aggregate_metrics.py \
    --output-root outputs/external/tc_qformer \
    --tasks analysis-0h forecast-6h forecast-12h
```

---

## Stage 6 — Regenerate the paper figures

The figures are driven by the eight CSVs under
`06_paper_figures/source_data/`, so they regenerate even without training
outputs:

```bash
python 06_paper_figures/01_make_paper_figures.py
```

To rebuild the source CSVs themselves (after re-training the baselines),
follow the helper docstrings inside `01_make_paper_figures.py`.

---

## Stage 7 — Optional: Ascend NPU full pipeline

If you have access to a Huawei Ascend NPU server, the entire pipeline can
be driven by:

```bash
bash 05_reproduction/ascend/env_check.sh       # sanity-check the toolkit
bash 05_reproduction/ascend/run_full_pipeline.sh
```

See `docs/06_ASCEND_REMOTE.md` for the override env-vars and the
single-card / DDP variants.

---

## 8. Determinism notes

- Set `PYTHONHASHSEED=0` and pass `--seed` (default 42) to every training
  script if you need bit-exact reproducibility on a single device.
- On Ascend NPU the order of float-reduction operations is not guaranteed
  to match CUDA; expect MAE differences on the order of 0.05 hPa across
  devices.
- The metadata baseline is deterministic given a fixed seed and a stable
  installed version of `xgboost` (any release in the `>=2.0` range that
  satisfies `pyproject.toml` works).
