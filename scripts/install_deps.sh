#!/usr/bin/env bash
set -euo pipefail

DIFFUSERS_VERSION="${DIFFUSERS_VERSION:-0.36.0}"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "ERROR: A virtual environment is active: ${VIRTUAL_ENV}"
  echo "Please run 'deactivate' first. This lab must use the Radeon Cloud base Python because it already contains ROCm PyTorch."
  exit 1
fi

if command -v git >/dev/null 2>&1 && [ -f /etc/ssl/certs/ca-certificates.crt ]; then
  echo "[1/5] git and CA certificates are already available; skipping apt refresh."
  git config --global http.sslCAInfo /etc/ssl/certs/ca-certificates.crt
elif command -v apt-get >/dev/null 2>&1; then
  echo "[1/5] Installing CA certificates and git if needed..."
  apt-get update || echo "WARNING: apt-get update failed partially; continuing because this lab does not need ROCm apt packages."
  apt-get install -y ca-certificates git || echo "WARNING: could not install ca-certificates/git; continuing with the current image."
  update-ca-certificates || true
  if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    git config --global http.sslCAInfo /etc/ssl/certs/ca-certificates.crt
  fi
else
  echo "[1/5] apt-get not found; skipping system CA refresh."
fi

echo "[2/5] Checking ROCm PyTorch from the Radeon Cloud base Python..."
python - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print("ERROR: PyTorch is not available in the base Python environment.")
    print("Please use a Radeon Cloud ROCm/PyTorch template or the teacher's prebuilt image.")
    print("Original import error:", repr(exc))
    sys.exit(1)

print("PyTorch:", torch.__version__)
print("PyTorch path:", torch.__file__)
print("CUDA/ROCm interface available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: No GPU is visible through torch.cuda.")
    print("Please check that the Radeon Cloud instance has GPU resources.")
    sys.exit(1)

print("GPU:", torch.cuda.get_device_name(0))
PY

TORCH_VERSION="$(python - <<'PY'
import torch
print(torch.__version__)
PY
)"
CONSTRAINT_FILE="$(mktemp)"
trap 'rm -f "${CONSTRAINT_FILE}"' EXIT
printf 'torch==%s\n' "${TORCH_VERSION}" > "${CONSTRAINT_FILE}"

echo "[3/5] Installing Python dependencies without replacing ROCm PyTorch..."
python -m pip install -U pip
python -m pip install -U --upgrade-strategy only-if-needed -c "${CONSTRAINT_FILE}" \
  "modelscope" \
  "accelerate" \
  "transformers>=4.56,<5" \
  "sentencepiece" \
  "safetensors" \
  "pillow"

echo "[4/5] Installing diffusers ${DIFFUSERS_VERSION} with Z-Image support from PyPI..."
python -m pip install -U --upgrade-strategy only-if-needed -c "${CONSTRAINT_FILE}" "diffusers==${DIFFUSERS_VERSION}"

echo "[5/5] Verifying environment..."
python - <<'PY'
import torch
from diffusers import ZImagePipeline
from modelscope import snapshot_download

print("PyTorch:", torch.__version__)
print("CUDA/ROCm interface available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Diffusers ZImagePipeline:", ZImagePipeline.__name__)
print("ModelScope snapshot_download:", snapshot_download.__name__)
PY

echo
echo "Environment is ready. Run the experiment scripts with the normal python command, for example:"
echo "  python scripts/01_generate_zimage.py --height 512 --width 512"
echo
echo "For the optional Gradio web app, first check whether Gradio is already available:"
echo "  python -c \"import gradio as gr; print(gr.__version__)\""
