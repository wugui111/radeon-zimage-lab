# Radeon Cloud Z-Image-Turbo Scripts

Recommended order:

1. `bash scripts/install_deps.sh`
2. `source .venv/bin/activate`
3. `python scripts/00_check_gpu.py`
4. `python scripts/01_generate_zimage.py`
5. `python scripts/02_gradio_zimage_app.py`
6. Optional advanced project: prepare LoRA data with `python scripts/03_prepare_lora_dataset.py`

Inside Radeon Cloud, PyTorch ROCm usually exposes the GPU through the `cuda` interface name. Seeing `cuda` in the scripts is expected.

`install_deps.sh` creates an isolated `.venv` with `--system-site-packages`, so it can reuse the platform ROCm/PyTorch installation without changing the base vLLM environment.

The script checks that ROCm PyTorch is already visible before installing dependencies. It installs `diffusers==0.36.0` from PyPI because this version already includes `ZImagePipeline` and avoids GitHub certificate failures during class.
