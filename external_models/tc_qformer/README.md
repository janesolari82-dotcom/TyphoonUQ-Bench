# TC-QFormer on TyphoonUQ-Bench

This directory hosts a worked adapter that wraps the TC-QFormer model of
Guo et al., *Interval-Based Tropical Cyclone Intensity Forecasting with
Spatiotemporal Transformers* (Remote Sensing 18(7):1069, 2026), into the
TyphoonUQ-Bench evaluation interface. After training, the script emits a
`predictions.parquet` file that the benchmark's evaluation entry point
(`04_evaluation/01_evaluate_predictions.py`) can score without any further
modification.

The adapter is provided as **one concrete example** of how an external
satellite-imagery UQ model can be plugged into the benchmark. It is not
intended as a universal template — methods built on reanalysis fields or
purely tabular inputs will require different glue code.

---

## 1. What was adapted

| Aspect                  | TC-QFormer (original)                          | TyphoonUQ-Bench                                | Adapter behaviour |
|-------------------------|------------------------------------------------|------------------------------------------------|-------------------|
| Source dataset          | TCIR (IR + WV, global, 2003–2016)              | Digital Typhoon (IR, W. Pacific, 1988–2023)    | Reads via `TyphoonClipDataset`                     |
| Input channels          | IR + WV (C = 2)                                | Single-channel IR                              | `in_channels=1`; first conv is replaced            |
| Frame resolution        | 128 × 128                                       | 224 × 224                                       | Bilinear-resized to 128 × 128 inside `forward`     |
| History length          | 4 frames (12 h history, 3 h spacing)            | 6 frames (6 h history, 1 h spacing)             | `num_frames=6`; temporal positions extended        |
| Forecast horizon        | 8 steps (T+3 to T+24, 3 h spacing)              | 1 step (analysis-0h, forecast-6h, forecast-12h) | `horizon=1`, one task per training run             |
| Target variable         | MSW (kt)                                        | Consensus central pressure (hPa)               | `create_tabular_samples` already returns hPa       |
| UQ output               | Pinball loss on 9 quantiles                    | Per-sample point, lower, upper bounds          | Quantiles decoded to point / 80% interval          |

Everything else (two-stage L1 → multi-quantile training, AdamW, OneCycle on
stage 1, fixed LR on stage 2, weight decay 1e-2) follows the paper's
defaults.

---

## 2. Files

| File | Purpose |
|------|---------|
| `model.py`               | TC-QFormer architecture (vision transformer + tabular fusion + quantile head). |
| `train_tc_qformer.py`    | End-to-end training entry point. Reads the benchmark manifest, runs both stages, and writes `predictions.parquet` + `metrics.json`. |
| `aggregate_metrics.py`   | Joins per-task `benchmark_metrics.json` into `summary.json` / `summary.csv`. |
| `run_tc_qformer_npu.sh`  | Driver script for Huawei Ascend NPU (sources the Ascend toolkit, then loops over the three tasks). |

---

## 3. Quick start (any device)

The adapter runs on CPU, CUDA, or Ascend NPU. Choose with `--device`.

Prerequisites (run from the repository root):

```bash
# 1. Install the benchmark package in editable mode and the ML extras.
pip install -e .[ml]

# 2. Make sure the benchmark manifest and split file exist.
#    Either run 02_benchmark_construction/01_build_aligned_manifest.py and
#    02_benchmark_construction/02_make_forward_splits.py, or download both from the Zenodo release (see Zenodo/README.md).
ls data/processed/benchmark/benchmark_manifest.parquet
ls outputs/splits/forward_main.parquet

# 3. Train + evaluate one task on CPU as a smoke test.
python external_models/tc_qformer/train_tc_qformer.py \
    --manifest data/processed/benchmark/benchmark_manifest.parquet \
    --split-file outputs/splits/forward_main.parquet \
    --data-root data/raw/digital_typhoon/archive \
    --task analysis-0h \
    --image-backend png \
    --device cpu \
    --batch-size 2 \
    --num-workers 0 \
    --stage1-epochs 1 --stage2-epochs 1 \
    --max-train-batches 5 --max-eval-batches 2 \
    --output-dir outputs/external/tc_qformer/smoke
```

A successful smoke run prints per-batch losses, writes
`outputs/external/tc_qformer/smoke/predictions.parquet`, and runs the
benchmark metrics in-process.

---

## 4. Full-scale training on Ascend NPU

`run_tc_qformer_npu.sh` orchestrates the full reproduction across the three
tasks. Override anything via environment variables:

```bash
PROJECT_ROOT=/path/to/this/repo \
ARCHIVE_ROOT=/path/to/digital_typhoon/archive \
BATCH_SIZE=8 NUM_WORKERS=0 \
STAGE1_EPOCHS=15 STAGE2_EPOCHS=35 \
bash external_models/tc_qformer/run_tc_qformer_npu.sh
```

The script writes one folder per task under `outputs/external/tc_qformer/`
and finishes by calling `aggregate_metrics.py` to produce
`summary.json` / `summary.csv`.

> **Ascend prerequisite.** Source the Ascend toolkit before launching
> (`source /usr/local/Ascend/ascend-toolkit/set_env.sh`). The driver
> script does this automatically when
> `ASCEND_ENV_SCRIPT` resolves to a real file; otherwise it exits with a
> helpful error.

---

## 5. Output contract

`predictions.parquet` produced by the adapter follows the same schema as
the built-in baselines and is therefore consumable by
`04_evaluation/01_evaluate_predictions.py`:

| Column        | Type    | Meaning                                                  |
|---------------|---------|----------------------------------------------------------|
| `sample_id`   | string  | Stable per-sample key joining back to the manifest.       |
| `target`      | float64 | Consensus central pressure (hPa).                         |
| `prediction`  | float64 | Point estimate (hPa).                                     |
| `lower`       | float64 | Lower bound of the model-native interval (hPa).           |
| `upper`       | float64 | Upper bound of the model-native interval (hPa).           |
| `lower_90`    | float64 | Lower bound after symmetric residual conformal (hPa).     |
| `upper_90`    | float64 | Upper bound after symmetric residual conformal (hPa).     |
| `split`       | string  | One of `train` / `val` / `test`.                          |
| `backbone`    | string  | Always `tc_qformer` for this adapter.                     |
| `device`      | string  | `cpu`, `cuda`, or `npu`.                                  |
| `storm_id`, `timestamp`, `task`, `target_lower`, `target_upper`, `pressure_range`, `agency_count`, `pressure_<agency>` | — | Sample metadata merged from the manifest. |

---

## 6. Common failures

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: No module named 'typhoonuq'` | Package not installed | Run `pip install -e .[ml]` from the repository root. |
| `Requested --device npu but torch.npu is unavailable` | Ascend toolkit not sourced | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` before launching. |
| `FileNotFoundError: predictions.parquet not found` (in driver script) | Training crashed silently | Re-run with `set -e` enabled (the shipped script already does this) and inspect `train.log`. |
| Very slow training on NFS storage | `num_workers > 0` thrashing NFS metadata | Set `--num-workers 0` and pin batch size to what the NPU can handle. |

---

## 7. Citation

If this adapter is useful, please cite both the TC-QFormer paper and the
TyphoonUQ-Bench benchmark.

```bibtex
@article{guo2026tcqformer,
  title     = {Interval-Based Tropical Cyclone Intensity Forecasting with Spatiotemporal Transformers},
  author    = {Guo, Tao and Zhang, Hao and Song, Tao and Peng, Shaoliang},
  journal   = {Remote Sensing},
  volume    = {18},
  number    = {7},
  pages     = {1069},
  year      = {2026},
}
```
