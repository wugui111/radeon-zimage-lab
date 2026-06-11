# Radeon Cloud Z-Image-Turbo Scripts

Recommended order:

1. `bash scripts/install_deps.sh`
2. `python scripts/01_generate_zimage.py`
3. `python scripts/02_gradio_zimage_app.py`
4. Optional advanced project: prepare LoRA data with `python scripts/03_prepare_lora_dataset.py`

Inside Radeon Cloud, PyTorch ROCm usually exposes the GPU through the `cuda` interface name. Seeing `cuda` in the scripts is expected.

