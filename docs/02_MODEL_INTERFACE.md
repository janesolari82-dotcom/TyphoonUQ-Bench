# Model interface and evaluation protocol

TyphoonUQ-Bench exposes two integration paths for external uncertainty-quantification models. Pick the lighter one when your model only needs tabular features; pick the heavier one when it needs the original satellite frames or the full PyTorch training stack.

---

## 1. Two integration paths at a glance

| Aspect | Lightweight path (§2) | Full path (§3) |
|--------|-----------------------|----------------|
| Recommended for | Tabular-only models; quick sanity checks; non-PyTorch ML stacks | Image-based UQ models; reproductions of prior work |
| Input format | Per-task **sample table parquet** built once by `02_benchmark_construction/03_build_task_samples.py` | `TyphoonClipDataset` PyTorch DataLoader |
| Required output columns | `sample_id`, `prediction` | `sample_id`, `target`, `prediction`, `lower`, `upper`, `split` |
| Evaluator | `04_evaluation/02_evaluate_external_predictions.py` | `04_evaluation/01_evaluate_predictions.py` |
| Example | `examples/01_custom_model_template_sklearn.py` (RandomForest, ~30 lines) | `examples/02_custom_model_template_torch.py`; full demo in `external_models/tc_qformer/` |

---

## 2. Lightweight path — sample-table interface

### 2.1 Build the sample tables once

```bash
python 02_benchmark_construction/03_build_task_samples.py \
    --manifest   data/processed/benchmark/benchmark_manifest.parquet \
    --split-file outputs/splits/forward_main.parquet \
    --image-backend png \
    --history-hours 6 \
    --output-dir outputs/splits/sample_cache
```

Each output parquet (`outputs/splits/sample_cache/<task>_png_h6_samples.parquet`) has one row per sample with columns:

| Column                  | Type    | Notes |
|-------------------------|---------|-------|
| `sample_id`             | string  | Unique evaluation key. |
| `split`                 | string  | `train`, `val`, or `test`. |
| `target`                | float   | Consensus central pressure (hPa). |
| `target_lower`, `target_upper` | float | Inter-agency pressure spread bounds (hPa). |
| `pressure_range`        | float   | `target_upper - target_lower`. |
| `agency_count`          | int     | Number of agencies contributing to consensus. |
| `storm_id`, `timestamp`, `task` | — | Metadata. |
| `image_refs`            | string  | Pipe-separated 6-frame image-path history. |
| `png_refs`, `h5_refs`   | string  | Backend-specific frame references. |
| `<feature>_t-<k>`       | float   | Tabular history features (8 features × 6 time steps = 48 columns). |
| `pressure_<agency>`     | float   | Per-agency pressure at the target time (when available). |

### 2.2 Produce a predictions file

External predictions are saved as CSV or parquet with at minimum:

| Column       | Type   | Required | Description |
|--------------|--------|----------|-------------|
| `sample_id`  | string | yes      | Must match the value in the sample table. |
| `prediction` | float  | yes      | Point estimate of consensus central pressure (hPa). |
| `lower`      | float  | optional | Lower bound (hPa). Defaults to `prediction` when absent. |
| `upper`      | float  | optional | Upper bound (hPa). Defaults to `prediction` when absent. |
| `lower_90`, `upper_90` | float | optional | 90% interval bounds for `range_coverage_at_90`. |

If only `prediction` is present, the evaluator treats it as a zero-width interval (so interval-width and coverage degrade gracefully).

### 2.3 Score the file

```bash
python 04_evaluation/02_evaluate_external_predictions.py \
    --samples     outputs/splits/sample_cache/forecast-6h_png_h6_samples.parquet \
    --predictions outputs/custom_model_predictions.parquet \
    --split       test \
    --output-json outputs/custom_model_metrics.json
```

The evaluator joins predictions to labels and metadata by `sample_id`, restricts to the requested split (default `test`), and reports all nine benchmark metrics. See `examples/01_custom_model_template_sklearn.py` for a 30-line scikit-learn worked example.

---

## 3. Full path — image + tabular DataLoader

When a model needs the raw satellite frames or wants to share the benchmark's PyTorch training stack, plug in at the `TyphoonClipDataset` level. Required output schema:

| Column       | Type    | Description |
|--------------|---------|-------------|
| `sample_id`  | string  | Per-sample key from `create_tabular_samples`. |
| `target`     | float   | Consensus central pressure (hPa) — copy through. |
| `prediction` | float   | Point estimate (hPa). |
| `lower`      | float   | Lower bound of the prediction interval (hPa). |
| `upper`      | float   | Upper bound of the prediction interval (hPa). |
| `split`      | string  | `train`, `val`, or `test`. |

Optional columns (`lower_90`, `upper_90`, `backbone`, `device`, plus range-aware metadata `target_lower`, `target_upper`, `pressure_range`, `agency_count`) unlock additional scoring. The built-in baselines and the TC-QFormer adapter auto-populate these.

Reference paths: `examples/02_custom_model_template_torch.py` is a DataLoader skeleton you copy and edit — the `predict_one_batch` function is the only thing you normally need to rewrite. `external_models/tc_qformer/` is a complete worked adapter for TC-QFormer (Guo et al. 2026) reusing the entire training stack.

Score with the built-in evaluator:

```bash
python 04_evaluation/01_evaluate_predictions.py \
    --predictions outputs/external/my_model/analysis-0h/predictions.parquet \
    --target-coverage 0.8 \
    --output-json outputs/external/my_model/analysis-0h/eval_metrics.json
```

---

## 4. Validating a submission

For either path, run the schema check before invoking the evaluator:

```bash
python examples/03_evaluate_submission.py \
    --predictions outputs/custom_model_predictions.parquet
```

It prints `[OK]` and exits 0 when the file conforms; otherwise it lists the failing columns and exits non-zero.

---

## 5. The evaluation suite

`src/typhoonuq/metrics.py::evaluate_predictions` returns one dict per call. Pure NumPy / Pandas implementation, negligible overhead on millions of rows.

### 5.1 Point metrics

| Key | Formula | Notes |
|-----|---------|-------|
| `mae`  | mean( \|target − prediction\| ) | Symmetric, in hPa. |
| `rmse` | sqrt( mean( (target − prediction)² ) ) | Penalises large residuals more. |

### 5.2 Interval metrics

Computed at the configured `target_coverage` (default 0.80; pass `--target-coverage` to override).

| Key | Description |
|-----|-------------|
| `interval_coverage` | Fraction of test rows with `lower ≤ target ≤ upper`. Ideal value = `target_coverage`. |
| `interval_width`    | Mean Prediction Interval Width (MPIW) in hPa. Lower is sharper. |
| `calibration_error` | Absolute discrepancy between empirical and nominal coverage at the configured `target_coverage` level: \|empirical_coverage − target_coverage\|. |

### 5.3 Range-aware metrics

Require `target_lower` / `target_upper`; `dispersion_interval_corr` additionally requires a non-NaN `pressure_range` column (i.e. rows where both qualifying agencies reported, so `pressure_range` carries a meaningful spread value).

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

## 6. Reference adapter

`external_models/tc_qformer/` ships a complete worked adapter that wraps TC-QFormer (Guo et al. 2026, *Remote Sensing* 18(7):1069) into the full integration path. It reads the same manifest and split file as the built-in baselines, runs the paper's two-stage training, and writes a `predictions.parquet` that the evaluator consumes directly. See `external_models/tc_qformer/README.md` for the full reproduction recipe.

TC-QFormer is the closest paradigm match for our benchmark (satellite imagery + quantile UQ). Adapters for methods built on reanalysis fields or pure best-track tabular inputs will need different glue code on the data-loading side and are planned for subsequent releases.
