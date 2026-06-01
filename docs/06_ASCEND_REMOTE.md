# Ascend Remote Reproduction

This optional workflow runs the torch baselines on a Linux server with Huawei
Ascend NPUs. The main benchmark contract is unchanged: the same manifest,
split file, sample cache, prediction schema, and metrics are used.

## Prerequisites

- A working CANN installation.
- A compatible `torch` and `torch_npu` stack.
- `npu-smi` and, for distributed runs, `torchrun` available on `PATH`.
- Raw data placed under the repository-relative paths documented in
  `docs/01_DATA.md`.

The default environment script is:

```bash
/usr/local/Ascend/ascend-toolkit/set_env.sh
```

Override it when needed:

```bash
export ASCEND_ENV_SCRIPT=/path/to/set_env.sh
```

## Environment Check

```bash
bash 05_reproduction/ascend/env_check.sh
```

## Full Pipeline

```bash
bash 05_reproduction/ascend/run_full_pipeline.sh
```

By default, this script writes derived data under `data/processed` and model
outputs under `outputs`.

Common overrides:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export TASKS="analysis-0h forecast-6h forecast-12h"
export IMAGE_BACKEND=png
export TEMPORAL_BACKEND=auto
export EPOCHS=3
export BATCH_SIZE=4
```

## Torch-Only Rerun

After the manifest and splits already exist:

```bash
bash 05_reproduction/ascend/run_torch_ddp.sh
```

For independent single-card jobs:

```bash
bash 05_reproduction/ascend/run_torch_independent_pool.sh
```

For a watcher that waits for free NPUs:

```bash
bash 05_reproduction/ascend/wait_for_free_npu_and_run_torch.sh
```

All scripts use repository-relative paths by default. Override `PROJECT_ROOT`,
`MANIFEST_PATH`, `SPLIT_FILE`, `ARCHIVE_ROOT`, or `OUTPUT_ROOT` only when you
intentionally run from a different layout.
