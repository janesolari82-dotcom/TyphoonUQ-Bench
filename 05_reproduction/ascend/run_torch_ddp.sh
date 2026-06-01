#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ASCEND_ENV_SCRIPT="${ASCEND_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

MANIFEST_PATH="${MANIFEST_PATH:-data/processed/benchmark/benchmark_manifest.parquet}"
SPLIT_FILE="${SPLIT_FILE:-outputs/splits/forward_main.parquet}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-data/raw/digital_typhoon/archive}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

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
echo "[INFO] MANIFEST_PATH=${MANIFEST_PATH}"
echo "[INFO] SPLIT_FILE=${SPLIT_FILE}"
echo "[INFO] ARCHIVE_ROOT=${ARCHIVE_ROOT}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "[INFO] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[INFO] TEMPORAL_BACKEND=${TEMPORAL_BACKEND}"
echo "[INFO] OUTPUT_NAMESPACE=${OUTPUT_NAMESPACE}"

output_family_dir() {
  local mode="$1"
  local backbone="$2"

  if [[ "${mode}" == "image-only" && "${backbone}" == "tiny" ]]; then
    echo "torch_image_${OUTPUT_NAMESPACE}"
    return
  fi
  if [[ "${mode}" == "multimodal" && "${backbone}" == "tiny" ]]; then
    echo "torch_multimodal_${OUTPUT_NAMESPACE}"
    return
  fi
  if [[ "${mode}" == "image-only" && "${backbone}" == "resnet18" ]]; then
    echo "torch_image_resnet18_${OUTPUT_NAMESPACE}"
    return
  fi
  if [[ "${mode}" == "multimodal" && "${backbone}" == "resnet18" ]]; then
    echo "torch_multimodal_resnet18_${OUTPUT_NAMESPACE}"
    return
  fi

  echo "[ERROR] Unsupported mode/backbone combination: ${mode} | ${backbone}" >&2
  exit 1
}

run_torch_job() {
  local task="$1"
  local mode="$2"
  local backbone="$3"
  local output_dir="$4"

  local extra_flags=()
  if [[ "${backbone}" == "resnet18" && "${USE_PRETRAINED}" == "1" ]]; then
    extra_flags+=(--pretrained)
  fi

  mkdir -p "${output_dir}"

  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    torchrun \
      --nproc_per_node "${NPROC_PER_NODE}" \
      --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" \
      03_model_training/02_train_torch_baseline.py \
      --manifest "${MANIFEST_PATH}" \
      --split-file "${SPLIT_FILE}" \
      --task "${task}" \
      --mode "${mode}" \
      --probabilistic gaussian \
      --backbone "${backbone}" \
      --device npu \
      --temporal-backend "${TEMPORAL_BACKEND}" \
      --num-workers "${NUM_WORKERS}" \
      --image-backend "${IMAGE_BACKEND}" \
      --data-root "${ARCHIVE_ROOT}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --learning-rate "${LEARNING_RATE}" \
      --output-dir "${output_dir}" \
      "${extra_flags[@]}"
  else
    python 03_model_training/02_train_torch_baseline.py \
      --manifest "${MANIFEST_PATH}" \
      --split-file "${SPLIT_FILE}" \
      --task "${task}" \
      --mode "${mode}" \
      --probabilistic gaussian \
      --backbone "${backbone}" \
      --device npu \
      --temporal-backend "${TEMPORAL_BACKEND}" \
      --num-workers "${NUM_WORKERS}" \
      --image-backend "${IMAGE_BACKEND}" \
      --data-root "${ARCHIVE_ROOT}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --learning-rate "${LEARNING_RATE}" \
      --output-dir "${output_dir}" \
      "${extra_flags[@]}"
  fi
}

for TASK in ${TASKS}; do
  if [[ "${RUN_TINY}" == "1" && "${RUN_IMAGE_ONLY}" == "1" ]]; then
    run_torch_job "${TASK}" image-only tiny "${OUTPUT_ROOT}/$(output_family_dir image-only tiny)/${TASK}"
  fi
  if [[ "${RUN_TINY}" == "1" && "${RUN_MULTIMODAL}" == "1" ]]; then
    run_torch_job "${TASK}" multimodal tiny "${OUTPUT_ROOT}/$(output_family_dir multimodal tiny)/${TASK}"
  fi
  if [[ "${RUN_RESNET18}" == "1" && "${RUN_IMAGE_ONLY}" == "1" ]]; then
    run_torch_job "${TASK}" image-only resnet18 "${OUTPUT_ROOT}/$(output_family_dir image-only resnet18)/${TASK}"
  fi
  if [[ "${RUN_RESNET18}" == "1" && "${RUN_MULTIMODAL}" == "1" ]]; then
    run_torch_job "${TASK}" multimodal resnet18 "${OUTPUT_ROOT}/$(output_family_dir multimodal resnet18)/${TASK}"
  fi
done
