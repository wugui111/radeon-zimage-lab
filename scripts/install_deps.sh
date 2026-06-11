#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"

if command -v apt-get >/dev/null 2>&1; then
  echo "[1/5] Installing CA certificates for GitHub/HTTPS access..."
  apt-get update
  apt-get install -y ca-certificates git
  update-ca-certificates
  if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    git config --global http.sslCAInfo /etc/ssl/certs/ca-certificates.crt
  fi
else
  echo "[1/5] apt-get not found; skipping system CA installation."
fi

echo "[2/5] Creating an isolated Python environment: ${VENV_DIR}"
python -m venv --system-site-packages "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[3/5] Installing Python dependencies in ${VENV_DIR}"
python -m pip install -U pip
python -m pip install -U \
  "modelscope" \
  "accelerate" \
  "transformers>=4.56,<5" \
  "sentencepiece" \
  "safetensors" \
  "gradio>=4.44,<6" \
  "pillow" \
  "starlette>=0.30,<1"

echo "[4/5] Installing latest diffusers with Z-Image support..."
python -m pip install -U "git+https://github.com/huggingface/diffusers"

echo "[5/5] Verifying environment..."
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA/ROCm interface available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "Environment is ready."
echo "Before running the experiment scripts, activate it with:"
echo "  source ${VENV_DIR}/bin/activate"
