#!/usr/bin/env bash
set -euo pipefail

# This is a reference entry point for the advanced project.
# Z-Image LoRA training is heavier than the 30-minute basic experiment.
# Check the latest DiffSynth-Studio Z-Image training guide before running in class:
# https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Z-Image.md

git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
python -m pip install -e .

echo "Prepare your dataset first, for example:"
echo "python ../03_prepare_lora_dataset.py --image-dir ./my_images --caption 'sks_person, realistic portrait photo, natural light, cinematic photography' --output-dir ./lora_dataset"
echo
echo "Then follow the current DiffSynth-Studio Z-Image LoRA command from the official document."
