# TyphoonUQ-Bench Architecture

## Layers

1. Raw data under the repository root
   - `data/raw/digital_typhoon/archive/`
   - `data/raw/jma/bst_all.txt`
   - `data/raw/ibtracs/ibtracs.WP.list.v04r01.csv`

2. Derived processed data
   - `data/processed/digital_typhoon/storm_index.parquet`
   - `data/processed/digital_typhoon/frame_index.parquet`
   - `data/processed/benchmark/jma_track.parquet`  *(intermediate artefact produced during manifest construction; not redistributed in the Zenodo bundle)*
   - `data/processed/benchmark/dt_jma_match_table.parquet`
   - `data/processed/benchmark/jma_ibtracs_match_table.parquet`
   - `data/processed/benchmark/benchmark_manifest.parquet`
   - `data/processed/benchmark/storm_match_table.parquet`
   - `data/processed/benchmark/coverage_report.parquet`
   - `data/processed/benchmark/manual_review.csv`

3. Python package
   - `src/typhoonuq/adapters/` parses the extracted Digital Typhoon archive
   - `src/typhoonuq/adapters/jma_best_track.py` parses the required JMA best track text
   - `src/typhoonuq/alignment.py` builds the JMA-anchored matching chain and final benchmark manifest
   - `src/typhoonuq/features.py` and `src/typhoonuq/datasets.py` create task samples
   - `src/typhoonuq/metrics.py` evaluates point and interval predictions
   - `src/typhoonuq/cli.py` exposes the installable workflow commands, including compatibility-preserving `tiny` and stronger `resnet18` torch backbones

4. Experiment outputs
   - `outputs/splits/`
   - `outputs/metadata/<task>/`
   - `outputs/torch_image/<task>/`
   - `outputs/torch_multimodal/<task>/`
   - recommended stronger torch output roots: `outputs/torch_image_resnet18/<task>/` and `outputs/torch_multimodal_resnet18/<task>/`
   - prediction tables include fixed `lower_90` / `upper_90` columns, target-time disagreement metadata, and optional model metadata columns such as `backbone` / `pretrained`

5. Release-facing interfaces
   - `README.md` is the authoritative runbook
   - `docs/03_REPRODUCE.md` documents the full 5×3 baseline grid and TC-QFormer reproduction recipe
   - `04_evaluation/03_run_case_study.py` is the storm-level case-study helper

## Design Rules

- Raw data is treated as immutable external input and is never rewritten in place.
- The repository ships no raw source data, generated benchmark tables, or training outputs.
- The cleaned upload package documents only the `png` training path.
- The full experiment contract is `Digital Typhoon -> indexes -> JMA anchor -> IBTrACS labels -> manifest -> splits -> metadata/image-only/multimodal baselines -> Haiyan case study`.
- Evaluation is disagreement-aware: standard point/UQ metrics remain intact, while prediction exports also support range hit scores, high-disagreement subset reporting, and per-agency control metrics.
- The original lightweight `tiny` torch models remain available for backward compatibility; the stronger `resnet18` route improves reference baselines without changing the benchmark definition or task semantics.
