#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUN_SCRIPT="${RUN_SCRIPT:-05_reproduction/ascend/run_torch_ddp.sh}"

POLL_SECONDS="${POLL_SECONDS:-30}"
MIN_FREE_NPUS="${MIN_FREE_NPUS:-1}"
CANDIDATE_DEVICES="${CANDIDATE_DEVICES:-0,1,2,3,4,5,6,7}"
CLAIM_LOCK_DIR="${CLAIM_LOCK_DIR:-logs/auto_npu_claim.lock}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/formal_retry}"
MANIFEST_PATH="${MANIFEST_PATH:-data/processed/benchmark/benchmark_manifest.parquet}"
SPLIT_FILE="${SPLIT_FILE:-${OUTPUT_ROOT}/splits/forward_main.parquet}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-data/raw/digital_typhoon/archive}"
PYDEPS_DIR="${PYDEPS_DIR:-.pydeps}"
ACL_OP_COMPILER_CACHE_MODE="${ACL_OP_COMPILER_CACHE_MODE:-enable}"
ACL_OP_COMPILER_CACHE_DIR="${ACL_OP_COMPILER_CACHE_DIR:-.cache/ascend_atc}"

cd "${PROJECT_ROOT}"
mkdir -p "logs"

if [[ ! -x "${RUN_SCRIPT}" && ! -f "${RUN_SCRIPT}" ]]; then
  echo "[ERROR] Run script not found: ${RUN_SCRIPT}" >&2
  exit 1
fi

if ! command -v npu-smi >/dev/null 2>&1; then
  echo "[ERROR] npu-smi not found in PATH" >&2
  exit 1
fi

if ! [[ "${MIN_FREE_NPUS}" =~ ^[0-9]+$ ]] || [[ "${MIN_FREE_NPUS}" -lt 1 ]]; then
  echo "[ERROR] MIN_FREE_NPUS must be a positive integer, got: ${MIN_FREE_NPUS}" >&2
  exit 1
fi

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

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

read_candidates() {
  local raw="${CANDIDATE_DEVICES//,/ }"
  local device
  CANDIDATE_LIST=()
  for device in ${raw}; do
    device="$(trim "${device}")"
    if [[ -n "${device}" ]]; then
      CANDIDATE_LIST+=("${device}")
    fi
  done
}

collect_occupied_devices() {
  local smi_output="$1"
  local line npu_field pid_field device_id
  OCCUPIED_LIST=()
  while IFS= read -r line; do
    [[ "${line}" == *"|"* ]] || continue
    npu_field="$(printf '%s' "${line}" | awk -F'|' '{print $2}')"
    pid_field="$(printf '%s' "${line}" | awk -F'|' '{print $3}')"
    npu_field="$(trim "${npu_field}")"
    pid_field="$(trim "${pid_field}")"
    [[ "${pid_field}" =~ ^[0-9]+$ ]] || continue
    device_id="${npu_field%% *}"
    [[ "${device_id}" =~ ^[0-9]+$ ]] || continue
    OCCUPIED_LIST+=("${device_id}")
  done <<< "${smi_output}"
}

is_occupied() {
  local needle="$1"
  local current
  for current in "${OCCUPIED_LIST[@]:-}"; do
    if [[ "${current}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

select_free_devices() {
  local candidate
  FREE_LIST=()
  for candidate in "${CANDIDATE_LIST[@]}"; do
    if ! is_occupied "${candidate}"; then
      FREE_LIST+=("${candidate}")
    fi
  done
}

claim_lock() {
  if mkdir "${CLAIM_LOCK_DIR}" 2>/dev/null; then
    trap 'rmdir "${CLAIM_LOCK_DIR}" 2>/dev/null || true' EXIT
    return 0
  fi
  return 1
}

launch_run() {
  local selected_csv
  selected_csv="$(IFS=,; printf '%s' "${SELECTED_LIST[*]}")"

  export PROJECT_ROOT
  export OUTPUT_ROOT
  export MANIFEST_PATH
  export SPLIT_FILE
  export ARCHIVE_ROOT
  export PYTHONPATH="${PYDEPS_DIR}:${PYTHONPATH:-}"
  export ACL_OP_COMPILER_CACHE_MODE
  export ACL_OP_COMPILER_CACHE_DIR
  export ASCEND_RT_VISIBLE_DEVICES="${selected_csv}"
  export NPROC_PER_NODE="${#SELECTED_LIST[@]}"

  echo "[INFO] Selected NPUs: ${ASCEND_RT_VISIBLE_DEVICES}"
  echo "[INFO] NPROC_PER_NODE=${NPROC_PER_NODE}"
  echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "[INFO] MANIFEST_PATH=${MANIFEST_PATH}"
  echo "[INFO] SPLIT_FILE=${SPLIT_FILE}"
  echo "[INFO] ARCHIVE_ROOT=${ARCHIVE_ROOT}"
  echo "[INFO] Launching: ${RUN_SCRIPT}"

  exec bash "${RUN_SCRIPT}"
}

read_candidates

echo "[INFO] Candidate devices: ${CANDIDATE_DEVICES}"
echo "[INFO] Required free NPUs: ${MIN_FREE_NPUS}"
echo "[INFO] Poll interval: ${POLL_SECONDS}s"
echo "[INFO] Waiting for free NPUs..."

while true; do
  SMI_OUTPUT="$(npu-smi info)"
  collect_occupied_devices "${SMI_OUTPUT}"
  select_free_devices

  if [[ "${#FREE_LIST[@]}" -ge "${MIN_FREE_NPUS}" ]]; then
    SELECTED_LIST=("${FREE_LIST[@]:0:${MIN_FREE_NPUS}}")
    if claim_lock; then
      echo "[INFO] Free NPUs found: $(IFS=,; printf '%s' "${FREE_LIST[*]}")"
      launch_run
    fi
    echo "[WARN] Another watcher already claimed the launch lock: ${CLAIM_LOCK_DIR}"
  else
    echo "[INFO] No enough free NPUs yet. Free now: ${#FREE_LIST[@]}/${MIN_FREE_NPUS}"
  fi

  sleep "${POLL_SECONDS}"
done
