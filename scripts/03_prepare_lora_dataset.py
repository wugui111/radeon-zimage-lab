import argparse
import csv
from pathlib import Path
from shutil import copy2


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a simple image-caption dataset for LoRA experiments.")
    parser.add_argument("--image-dir", required=True, help="Directory containing source images.")
    parser.add_argument("--caption", required=True, help="Default caption for every image.")
    parser.add_argument("--output-dir", default="lora_dataset", help="Output dataset directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_images = output_dir / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    image_files = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_files.extend(image_dir.glob(pattern))
    image_files = sorted(image_files)

    if not image_files:
        raise RuntimeError(f"No images found in {image_dir}")

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
        writer.writeheader()
        for idx, image_path in enumerate(image_files, start=1):
            target_name = f"image_{idx:03d}{image_path.suffix.lower()}"
            copy2(image_path, output_images / target_name)
            writer.writerow({"file_name": f"images/{target_name}", "text": args.caption})

    print(f"Prepared {len(image_files)} images.")
    print(f"Dataset directory: {output_dir.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()

