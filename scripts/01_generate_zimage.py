import argparse
import os
import time
from pathlib import Path

import torch
from diffusers import ZImagePipeline
from modelscope import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="Generate one image with Z-Image-Turbo on Radeon Cloud.")
    parser.add_argument("--prompt", type=str, default="为软件技术专业实训周设计一张中文宣传海报，主题是云计算与人工智能应用开发，蓝白科技风，清晰中文标题，现代设计，高质量")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/zimage_result.png")
    parser.add_argument("--cache-dir", type=str, default=os.environ.get("MODEL_CACHE_DIR", "./models"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU is available. Please run this script inside a Radeon Cloud GPU notebook.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Downloading or loading model cache...")
    model_dir = snapshot_download("Tongyi-MAI/Z-Image-Turbo", cache_dir=args.cache_dir)

    pipe = ZImagePipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator("cuda").manual_seed(args.seed)
    torch.cuda.synchronize()
    start = time.time()

    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=generator,
    ).images[0]

    torch.cuda.synchronize()
    elapsed = time.time() - start
    image.save(output_path)

    print(f"Image saved to: {output_path.resolve()}")
    print(f"Generation time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
