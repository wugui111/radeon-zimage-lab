import os

import torch
from diffusers import ZImagePipeline
from modelscope import snapshot_download


MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/opt/models")

print("Cache dir:", CACHE_DIR)
print("Downloading model from ModelScope if needed...")
model_dir = snapshot_download(MODEL_ID, cache_dir=CACHE_DIR)
print("Model dir:", model_dir)

print("Loading pipeline once to verify dependencies...")
pipe = ZImagePipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
print("Pipeline loaded:", type(pipe).__name__)

