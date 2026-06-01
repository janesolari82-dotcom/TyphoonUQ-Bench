#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ASCEND_ENV_SCRIPT="${ASCEND_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

if [[ ! -f "${ASCEND_ENV_SCRIPT}" ]]; then
  echo "[ERROR] Ascend environment script not found: ${ASCEND_ENV_SCRIPT}" >&2
  exit 1
fi

set +u
source "${ASCEND_ENV_SCRIPT}"
set -u

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] ASCEND_ENV_SCRIPT=${ASCEND_ENV_SCRIPT}"
echo "[INFO] ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"

if command -v npu-smi >/dev/null 2>&1; then
  echo "[INFO] npu-smi info"
  npu-smi info
else
  echo "[WARN] npu-smi not found in PATH"
fi

if command -v torchrun >/dev/null 2>&1; then
  echo "[INFO] torchrun available: $(command -v torchrun)"
else
  echo "[WARN] torchrun not found in PATH"
fi

python3 - <<'PY'
import importlib
import json
import sys

report = {
    "python": sys.version,
    "modules": {},
}

for name in ["torch", "torchvision", "torch_npu", "pandas", "pyarrow"]:
    try:
        module = importlib.import_module(name)
        report["modules"][name] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        report["modules"][name] = f"ERROR: {type(exc).__name__}: {exc}"

print(json.dumps(report, indent=2, ensure_ascii=False))

try:
    import torch
    has_npu = hasattr(torch, "npu")
    print(f"torch_has_npu={has_npu}")
    if has_npu:
        available = torch.npu.is_available()
        print(f"torch_npu_available={available}")
        if available:
            device = torch.device("npu:0")
            a = torch.randn(2, 2, device=device)
            b = torch.randn(2, 2, device=device)
            c = a @ b
            print(f"npu_mm_ok shape={tuple(c.shape)} device={c.device}")
except Exception as exc:
    print(f"npu_check_error={type(exc).__name__}: {exc}")
PY
