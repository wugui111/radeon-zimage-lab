# Radeon Cloud Z-Image-Turbo Scripts

Recommended order:

1. `bash scripts/install_deps.sh`
2. `python scripts/00_check_gpu.py`
3. `python scripts/01_generate_zimage.py`
4. `python scripts/02_gradio_zimage_app.py`
5. Optional advanced project: prepare LoRA data with `python scripts/03_prepare_lora_dataset.py`

Inside Radeon Cloud, PyTorch ROCm usually exposes the GPU through the `cuda` interface name. Seeing `cuda` in the scripts is expected.

`install_deps.sh` uses the Radeon Cloud base Python directly because this template's ROCm PyTorch is visible there but not reliably visible inside a normal `.venv`.

The script checks that ROCm PyTorch is already visible, constrains the current `torch` version so pip does not replace it, and installs `diffusers==0.36.0` from PyPI because this version already includes `ZImagePipeline` and avoids GitHub certificate failures during class.
