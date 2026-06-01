#!/usr/bin/env bash
# Run TC-QFormer reproduction on all three TyphoonUQ-Bench tracks on Ascend NPU.
#
# Usage (from project root):
#   bash external_models/tc_qformer/run_tc_qformer_npu.sh
#
# Overridable env vars:
#   PROJECT_ROOT          (default: auto-detected from script path)
#   ASCEND_ENV_SCRIPT     (default: /usr/local/Ascend/ascend-toolkit/set_env.sh)
#   MANIFEST_PATH         (default: $PROJECT_ROOT/data/processed/benchmark/benchmark_manifest.parquet)
#   SPLIT_FILE            (default: $PROJECT_ROOT/outputs/splits/forward_main.parquet)
#   ARCHIVE_ROOT          (default: $PROJECT_ROOT/data/raw/digital_typhoon/archive)
#   OUTPUT_ROOT           (default: $PROJECT_ROOT/outputs/external/tc_qformer)
#   IMAGE_BACKEND         (default: png)
#   ASCEND_RT_VISIBLE_DEVICES (default: 0)
#   BATCH_SIZE            (default: 8)
#   STAGE1_EPOCHS         (default: 15)
#   STAGE2_EPOCHS         (default: 35)
#   STAGE1_LR             (default: 5e-6)
#   STAGE2_LR             (default: 3e-6)
#   TASKS                 (default: "analysis-0h forecast-6h forecast-12h")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ASCEND_ENV_SCRIPT="${ASCEND_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

MANIFEST_PATH="${MANIFEST_PATH:-${PROJECT_ROOT}/data/processed/benchmark/benchmark_manifest.parquet}"
SPLIT_FILE="${SPLIT_FILE:-${PROJECT_ROOT}/outputs/splits/forward_main.parquet}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${PROJECT_ROOT}/data/raw/digital_typhoon/archive}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/external/tc_qformer}"

IMAGE_BACKEND="${IMAGE_BACKEND:-png}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-15}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-35}"
STAGE1_LR="${STAGE1_LR:-5e-6}"
STAGE2_LR="${STAGE2_LR:-3e-6}"
TASKS="${TASKS:-analysis-0h forecast-6h forecast-12h}"

if [[ ! -f "${ASCEND_ENV_SCRIPT}" ]]; then
  echo "[ERROR] Ascend environment script not found: ${ASCEND_ENV_SCRIPT}" >&2
  exit 1
fi

source "${ASCEND_ENV_SCRIPT}"
export ASCEND_RT_VISIBLE_DEVICES
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] MANIFEST_PATH=${MANIFEST_PATH}"
echo "[INFO] SPLIT_FILE=${SPLIT_FILE}"
echo "[INFO] ARCHIVE_ROOT=${ARCHIVE_ROOT}"
echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] IMAGE_BACKEND=${IMAGE_BACKEND}"
echo "[INFO] ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "[INFO] TASKS=${TASKS}"

for TASK in ${TASKS}; do
  TASK_OUT="${OUTPUT_ROOT}/${TASK}"
  mkdir -p "${TASK_OUT}"
  echo "[INFO] >>> training TC-QFormer for task=${TASK} -> ${TASK_OUT}"
  python external_models/tc_qformer/train_tc_qformer.py \
    --manifest "${MANIFEST_PATH}" \
    --split-file "${SPLIT_FILE}" \
    --data-root "${ARCHIVE_ROOT}" \
    --task "${TASK}" \
    --image-backend "${IMAGE_BACKEND}" \
    --device npu \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --stage1-epochs "${STAGE1_EPOCHS}" \
    --stage2-epochs "${STAGE2_EPOCHS}" \
    --stage1-lr "${STAGE1_LR}" \
    --stage2-lr "${STAGE2_LR}" \
    --output-dir "${TASK_OUT}" 2>&1 | tee "${TASK_OUT}/train.log"

  python 04_evaluation/01_evaluate_predictions.py \
    --predictions "${TASK_OUT}/predictions.parquet" \
    --target-coverage 0.8 \
    --output-json "${TASK_OUT}/benchmark_metrics.json"
done

echo "[INFO] aggregating per-task metrics"
python external_models/tc_qformer/aggregate_metrics.py \
  --output-root "${OUTPUT_ROOT}" \
  --tasks ${TASKS}
echo "[INFO] all done -> ${OUTPUT_ROOT}/summary.json"
