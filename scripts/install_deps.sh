#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip
python -m pip install -U modelscope accelerate transformers sentencepiece safetensors gradio pillow
python -m pip install -U "git+https://github.com/huggingface/diffusers"

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA/ROCm interface available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

