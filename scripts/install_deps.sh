#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
DIFFUSERS_VERSION="${DIFFUSERS_VERSION:-0.36.0}"

if command -v apt-get >/dev/null 2>&1; then
  echo "[1/6] Installing CA certificates for GitHub/HTTPS access..."
  apt-get update
  apt-get install -y ca-certificates git
  update-ca-certificates
  if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    git config --global http.sslCAInfo /etc/ssl/certs/ca-certificates.crt
  fi
else
  echo "[1/6] apt-get not found; skipping system CA installation."
fi

echo "[2/6] Creating an isolated Python environment: ${VENV_DIR}"
python -m venv --system-site-packages "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[3/6] Checking ROCm PyTorch from the platform image..."
if ! python - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print("ERROR: PyTorch is not available in this Python environment.")
    print("Please launch a Radeon Cloud template that already includes ROCm PyTorch, then rerun this script.")
    print("Original import error:", repr(exc))
    sys.exit(1)

print("PyTorch:", torch.__version__)
print("PyTorch path:", torch.__file__)
print("CUDA/ROCm interface available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: No GPU is visible through torch.cuda.")
    print("Do not continue installing PyPI torch; use a ROCm/PyTorch Radeon Cloud image instead.")
    sys.exit(1)

print("GPU:", torch.cuda.get_device_name(0))
PY
then
  echo
  echo "The current .venv cannot see ROCm PyTorch."
  echo "If this is a rerun after a failed install, remove the old environment first:"
  echo "  rm -rf ${VENV_DIR}"
  echo "Then start from a Radeon Cloud ROCm/PyTorch template and run this script again."
  exit 1
fi

TORCH_VERSION="$(python - <<'PY'
import torch
print(torch.__version__)
PY
)"
CONSTRAINT_FILE="$(mktemp)"
trap 'rm -f "${CONSTRAINT_FILE}"' EXIT
printf 'torch==%s\n' "${TORCH_VERSION}" > "${CONSTRAINT_FILE}"

echo "[4/6] Installing Python dependencies in ${VENV_DIR}"
python -m pip install -U pip
python -m pip install -U --upgrade-strategy only-if-needed -c "${CONSTRAINT_FILE}" \
  "modelscope" \
  "accelerate" \
  "transformers>=4.56,<5" \
  "sentencepiece" \
  "safetensors" \
  "gradio>=4.44,<6" \
  "pillow" \
  "starlette>=0.30,<1"

echo "[5/6] Installing diffusers ${DIFFUSERS_VERSION} with Z-Image support from PyPI..."
python -m pip install -U --upgrade-strategy only-if-needed -c "${CONSTRAINT_FILE}" "diffusers==${DIFFUSERS_VERSION}"

echo "[6/6] Verifying environment..."
python - <<'PY'
import torch
from diffusers import ZImagePipeline

print("PyTorch:", torch.__version__)
print("CUDA/ROCm interface available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Diffusers ZImagePipeline:", ZImagePipeline.__name__)
PY

echo
echo "Environment is ready."
echo "Before running the experiment scripts, activate it with:"
echo "  source ${VENV_DIR}/bin/activate"
