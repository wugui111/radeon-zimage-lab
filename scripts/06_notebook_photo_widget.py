import os
import time
from pathlib import Path

from IPython.display import display

try:
    import ipywidgets as widgets
except Exception as exc:
    raise SystemExit(
        "ipywidgets is not available. Install it with: python -m pip install ipywidgets\n"
        f"Original import error: {exc!r}"
    )

import torch
from diffusers import ZImagePipeline
from modelscope import snapshot_download


MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "./models")

STYLE_PREFIX = {
    "人像写真": "真实摄影，人像写真，85mm镜头，浅景深，自然肤色，电影感光线，画面中不包含文字和水印，",
    "旅行风景照": "真实旅行摄影，广角镜头，自然风景，日出或日落光线，高细节，画面中不包含文字和水印，",
    "城市街拍": "真实城市街拍，35mm镜头，自然抓拍，街道环境光，电影色彩，画面中不包含文字和水印，",
    "户外运动照": "真实户外运动摄影，动态构图，清晰主体，自然光，高速快门质感，画面中不包含文字和水印，",
}


if not torch.cuda.is_available():
    raise RuntimeError("No GPU is available. Please run this notebook inside a Radeon Cloud GPU instance.")

print("GPU:", torch.cuda.get_device_name(0))
print("Loading model. The first run may take several minutes if the model is not cached.")
model_dir = snapshot_download(MODEL_ID, cache_dir=CACHE_DIR)
pipe = ZImagePipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
).to("cuda")
print("Model is ready.")


prompt = widgets.Textarea(
    value="一位年轻旅行者站在清晨的高山湖泊旁，远处雪山和薄雾，金色日出光线，自然表情",
    description="照片描述",
    layout=widgets.Layout(width="720px", height="90px"),
)
style = widgets.Dropdown(
    options=list(STYLE_PREFIX.keys()),
    value="旅行风景照",
    description="摄影风格",
)
size = widgets.Dropdown(
    options=["512x512", "768x768", "1024x1024"],
    value="768x768",
    description="图片尺寸",
)
seed = widgets.IntText(value=42, description="随机种子")
button = widgets.Button(description="生成图片", button_style="primary")
output = widgets.Output()


def generate(_):
    height, width = map(int, size.value.split("x"))
    final_prompt = STYLE_PREFIX[style.value] + prompt.value
    generator = torch.Generator("cuda").manual_seed(int(seed.value))

    with output:
        output.clear_output()
        print("Generating...")
        start = time.time()
        image = pipe(
            prompt=final_prompt,
            height=height,
            width=width,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        torch.cuda.synchronize()
        elapsed = time.time() - start

        Path("outputs").mkdir(exist_ok=True)
        output_path = Path("outputs") / f"notebook_widget_seed_{seed.value}.png"
        image.save(output_path)
        print(f"Saved to: {output_path.resolve()}")
        print(f"Generation time: {elapsed:.2f} seconds")
        display(image)


button.on_click(generate)
display(widgets.VBox([prompt, style, size, seed, button, output]))
