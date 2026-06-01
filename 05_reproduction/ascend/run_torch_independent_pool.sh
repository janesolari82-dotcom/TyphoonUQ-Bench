#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ASCEND_ENV_SCRIPT="${ASCEND_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

MANIFEST_PATH="${MANIFEST_PATH:-data/processed/benchmark/benchmark_manifest.parquet}"
SPLIT_FILE="${SPLIT_FILE:-outputs/splits/forward_main.parquet}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-data/raw/digital_typhoon/archive}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-20}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
IMAGE_BACKEND="${IMAGE_BACKEND:-png}"
TEMPORAL_BACKEND="${TEMPORAL_BACKEND:-auto}"
OUTPUT_NAMESPACE="${OUTPUT_NAMESPACE:-ascend}"
TASKS="${TASKS:-analysis-0h forecast-6h forecast-12h}"

RUN_TINY="${RUN_TINY:-1}"
RUN_RESNET18="${RUN_RESNET18:-1}"
RUN_IMAGE_ONLY="${RUN_IMAGE_ONLY:-1}"
RUN_MULTIMODAL="${RUN_MULTIMODAL:-1}"
USE_PRETRAINED="${USE_PRETRAINED:-0}"

PYDEPS_DIR="${PYDEPS_DIR:-.pydeps}"
ACL_OP_COMPILER_CACHE_MODE="${ACL_OP_COMPILER_CACHE_MODE:-enable}"
ACL_OP_COMPILER_CACHE_DIR="${ACL_OP_COMPILER_CACHE_DIR:-.cache/ascend_atc}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
POOL_TAG="${POOL_TAG:-pool_$(date +%F_%H%M%S)}"
POOL_LOG_DIR="${POOL_LOG_DIR:-logs/${POOL_TAG}}"

if [[ ! -f "${ASCEND_ENV_SCRIPT}" ]]; then
  echo "[ERROR] Ascend environment script not found: ${ASCEND_ENV_SCRIPT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "[ERROR] Manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

if [[ ! -f "${SPLIT_FILE}" ]]; then
  echo "[ERROR] Split file not found: ${SPLIT_FILE}" >&2
  exit 1
fi

if [[ ! -d "${ARCHIVE_ROOT}" ]]; then
  echo "[ERROR] Archive root not found: ${ARCHIVE_ROOT}" >&2
  exit 1
fi

set +u
source "${ASCEND_ENV_SCRIPT}"
set -u

export PYTHONPATH="${PYDEPS_DIR}:${PYTHONPATH:-}"
export ACL_OP_COMPILER_CACHE_MODE
export ACL_OP_COMPILER_CACHE_DIR
export OMP_NUM_THREADS

mkdir -p "${POOL_LOG_DIR}" "${OUTPUT_ROOT}"

echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] TEMPORAL_BACKEND=${TEMPORAL_BACKEND}"
echo "[INFO] OUTPUT_NAMESPACE=${OUTPUT_NAMESPACE}"
echo "[INFO] POOL_LOG_DIR=${POOL_LOG_DIR}"

IFS=',' read -r -a DEVICES <<< "${ASCEND_RT_VISIBLE_DEVICES}"

declare -a JOBS=()

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

append_job() {
  local task="$1"
  local mode="$2"
  local backbone="$3"
  local output_dir="$4"
  local pretrained_flag="$5"
  local done_flag="${output_dir}/metrics.json"

  if [[ -f "${done_flag}" ]]; then
    echo "[INFO] Skip completed job: ${task} | ${mode} | ${backbone}"
    return
  fi

  JOBS+=("${task}|${mode}|${backbone}|${output_dir}|${pretrained_flag}")
}

for TASK in ${TASKS}; do
  if [[ "${RUN_TINY}" == "1" && "${RUN_IMAGE_ONLY}" == "1" ]]; then
    append_job "${TASK}" "image-only" "tiny" "${OUTPUT_ROOT}/$(output_family_dir image-only tiny)/${TASK}" "0"
  fi
  if [[ "${RUN_TINY}" == "1" && "${RUN_MULTIMODAL}" == "1" ]]; then
    append_job "${TASK}" "multimodal" "tiny" "${OUTPUT_ROOT}/$(output_family_dir multimodal tiny)/${TASK}" "0"
  fi
  if [[ "${RUN_RESNET18}" == "1" && "${RUN_IMAGE_ONLY}" == "1" ]]; then
    append_job "${TASK}" "image-only" "resnet18" "${OUTPUT_ROOT}/$(output_family_dir image-only resnet18)/${TASK}" "${USE_PRETRAINED}"
  fi
  if [[ "${RUN_RESNET18}" == "1" && "${RUN_MULTIMODAL}" == "1" ]]; then
    append_job "${TASK}" "multimodal" "resnet18" "${OUTPUT_ROOT}/$(output_family_dir multimodal resnet18)/${TASK}" "${USE_PRETRAINED}"
  fi
done

if [[ "${#JOBS[@]}" -eq 0 ]]; then
  echo "[INFO] No pending torch jobs. Everything is already complete."
  exit 0
fi

declare -A PID_TO_DEVICE=()
declare -A PID_TO_LABEL=()
declare -a ACTIVE_PIDS=()
LAST_LAUNCHED_PID=""

launch_job() {
  local device="$1"
  local payload="$2"
  local task mode backbone output_dir pretrained_flag
  IFS='|' read -r task mode backbone output_dir pretrained_flag <<< "${payload}"

  mkdir -p "${output_dir}"
  local job_label="${task}__${mode}__${backbone}__${OUTPUT_NAMESPACE}__${TEMPORAL_BACKEND}__npu${device}"
  local log_file="${POOL_LOG_DIR}/${job_label}.log"
  local -a extra_flags=()

  if [[ "${backbone}" == "resnet18" && "${pretrained_flag}" == "1" ]]; then
    extra_flags+=(--pretrained)
  fi

  (
    export ASCEND_RT_VISIBLE_DEVICES="${device}"
    python3 03_model_training/02_train_torch_baseline.py \
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
    echo "[INFO] Completed ${job_label}"
  ) > "${log_file}" 2>&1 &

  local pid="$!"
  PID_TO_DEVICE["${pid}"]="${device}"
  PID_TO_LABEL["${pid}"]="${job_label}"
  LAST_LAUNCHED_PID="${pid}"
  echo "[INFO] Launched ${job_label} pid=${pid} log=${log_file}"
}

JOB_INDEX=0
FAIL_COUNT=0

for device in "${DEVICES[@]}"; do
  if [[ "${JOB_INDEX}" -ge "${#JOBS[@]}" ]]; then
    break
  fi
  launch_job "${device}" "${JOBS[${JOB_INDEX}]}"
  ACTIVE_PIDS+=("${LAST_LAUNCHED_PID}")
  JOB_INDEX=$((JOB_INDEX + 1))
done

while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
  declare -a NEXT_PIDS=()
  for pid in "${ACTIVE_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      NEXT_PIDS+=("${pid}")
      continue
    fi

    if wait "${pid}"; then
      echo "[INFO] Finished ${PID_TO_LABEL[${pid}]}"
    else
      echo "[ERROR] Failed ${PID_TO_LABEL[${pid}]}" >&2
      FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    if [[ "${JOB_INDEX}" -lt "${#JOBS[@]}" ]]; then
      launch_job "${PID_TO_DEVICE[${pid}]}" "${JOBS[${JOB_INDEX}]}"
      NEXT_PIDS+=("${LAST_LAUNCHED_PID}")
      JOB_INDEX=$((JOB_INDEX + 1))
    fi
  done
  ACTIVE_PIDS=("${NEXT_PIDS[@]}")
  sleep 5
done

if [[ "${FAIL_COUNT}" -gt 0 ]]; then
  echo "[ERROR] ${FAIL_COUNT} job(s) failed. Check logs under ${POOL_LOG_DIR}" >&2
  exit 1
fi

echo "[INFO] All independent torch jobs completed successfully."
