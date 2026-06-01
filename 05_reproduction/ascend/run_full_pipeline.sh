#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ASCEND_ENV_SCRIPT="${ASCEND_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

ARCHIVE_ROOT="${ARCHIVE_ROOT:-data/raw/digital_typhoon/archive}"
JMA_BEST_TRACK="${JMA_BEST_TRACK:-data/raw/jma/bst_all.txt}"
IBTRACS_CSV="${IBTRACS_CSV:-data/raw/ibtracs/ibtracs.WP.list.v04r01.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

DT_PROCESSED_DIR="${DT_PROCESSED_DIR:-data/processed/digital_typhoon}"
BENCHMARK_DIR="${BENCHMARK_DIR:-data/processed/benchmark}"
SPLIT_DIR="${SPLIT_DIR:-${OUTPUT_ROOT}/splits}"

ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
IMAGE_BACKEND="${IMAGE_BACKEND:-png}"
TEMPORAL_BACKEND="${TEMPORAL_BACKEND:-auto}"
OUTPUT_NAMESPACE="${OUTPUT_NAMESPACE:-ascend}"
TASKS="${TASKS:-analysis-0h forecast-6h forecast-12h}"

RUN_METADATA="${RUN_METADATA:-1}"
RUN_TINY="${RUN_TINY:-1}"
RUN_RESNET18="${RUN_RESNET18:-1}"
RUN_IMAGE_ONLY="${RUN_IMAGE_ONLY:-1}"
RUN_MULTIMODAL="${RUN_MULTIMODAL:-1}"
USE_PRETRAINED="${USE_PRETRAINED:-1}"

if [[ ! -f "${ASCEND_ENV_SCRIPT}" ]]; then
  echo "[ERROR] Ascend environment script not found: ${ASCEND_ENV_SCRIPT}" >&2
  exit 1
fi

source "${ASCEND_ENV_SCRIPT}"
export ASCEND_RT_VISIBLE_DEVICES

cd "${PROJECT_ROOT}"

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] ARCHIVE_ROOT=${ARCHIVE_ROOT}"
echo "[INFO] JMA_BEST_TRACK=${JMA_BEST_TRACK}"
echo "[INFO] IBTRACS_CSV=${IBTRACS_CSV}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] TEMPORAL_BACKEND=${TEMPORAL_BACKEND}"
echo "[INFO] OUTPUT_NAMESPACE=${OUTPUT_NAMESPACE}"

mkdir -p "${DT_PROCESSED_DIR}" "${BENCHMARK_DIR}" "${SPLIT_DIR}" "${OUTPUT_ROOT}"

python 01_data_preparation/01_build_digital_typhoon_index.py \
  --archive-root "${ARCHIVE_ROOT}" \
  --output-dir "${DT_PROCESSED_DIR}"

python 02_benchmark_construction/01_build_aligned_manifest.py \
  --storm-index "${DT_PROCESSED_DIR}/storm_index.parquet" \
  --frame-index "${DT_PROCESSED_DIR}/frame_index.parquet" \
  --jma-best-track "${JMA_BEST_TRACK}" \
  --ibtracs-csv "${IBTRACS_CSV}" \
  --output-dir "${BENCHMARK_DIR}"

python 02_benchmark_construction/02_make_forward_splits.py \
  --manifest "${BENCHMARK_DIR}/benchmark_manifest.parquet" \
  --output-dir "${SPLIT_DIR}"

if [[ "${RUN_METADATA}" == "1" ]]; then
  for TASK in ${TASKS}; do
    python 03_model_training/01_train_metadata_baseline.py \
      --manifest "${BENCHMARK_DIR}/benchmark_manifest.parquet" \
      --split-file "${SPLIT_DIR}/forward_main.parquet" \
      --task "${TASK}" \
      --image-backend "${IMAGE_BACKEND}" \
      --output-dir "${OUTPUT_ROOT}/metadata/${TASK}"
  done
fi

MANIFEST_PATH="${BENCHMARK_DIR}/benchmark_manifest.parquet" \
SPLIT_FILE="${SPLIT_DIR}/forward_main.parquet" \
ARCHIVE_ROOT="${ARCHIVE_ROOT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
NUM_WORKERS="${NUM_WORKERS}" \
BATCH_SIZE="${BATCH_SIZE}" \
EPOCHS="${EPOCHS}" \
LEARNING_RATE="${LEARNING_RATE}" \
IMAGE_BACKEND="${IMAGE_BACKEND}" \
TEMPORAL_BACKEND="${TEMPORAL_BACKEND}" \
OUTPUT_NAMESPACE="${OUTPUT_NAMESPACE}" \
TASKS="${TASKS}" \
RUN_TINY="${RUN_TINY}" \
RUN_RESNET18="${RUN_RESNET18}" \
RUN_IMAGE_ONLY="${RUN_IMAGE_ONLY}" \
RUN_MULTIMODAL="${RUN_MULTIMODAL}" \
USE_PRETRAINED="${USE_PRETRAINED}" \
bash "05_reproduction/ascend/run_torch_ddp.sh"
