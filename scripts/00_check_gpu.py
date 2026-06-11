import torch

print("PyTorch:", torch.__version__)
print("CUDA/ROCm interface available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("GPU memory:", round(props.total_memory / 1024**3, 2), "GB")
else:
    print("No GPU was detected. Please check that the Radeon Cloud instance has started with GPU resources.")

