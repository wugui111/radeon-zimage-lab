import os
import random
import time
from datetime import datetime
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
OUTPUT_DIR = Path("outputs")

STYLE_PREFIX = {
    "人像写真": "真实摄影，人像写真，85mm镜头，浅景深，自然肤色，电影感光线，画面中不包含文字和水印，",
    "旅行风景照": "真实旅行摄影，广角镜头，自然风景，日出或日落光线，高细节，画面中不包含文字和水印，",
    "城市街拍": "真实城市街拍，35mm镜头，自然抓拍，街道环境光，电影色彩，画面中不包含文字和水印，",
    "户外运动照": "真实户外运动摄影，动态构图，清晰主体，自然光，高速快门质感，画面中不包含文字和水印，",
}

PROMPT_PRESETS = {
    "高山湖泊旅行照": (
        "旅行风景照",
        "一位年轻旅行者站在清晨的高山湖泊旁，远处雪山和薄雾，金色日出光线，自然表情",
    ),
    "城市夜景街拍": (
        "城市街拍",
        "一位年轻人在雨后的城市街道散步，霓虹灯倒映在路面，电影感色彩，浅景深",
    ),
    "海边日落人像": (
        "人像写真",
        "一位人物站在海边日落时分，逆光轮廓，柔和金色光线，真实自然表情",
    ),
    "森林徒步照片": (
        "户外运动照",
        "一位徒步者走在清晨森林小路上，背包，阳光穿过树叶，空气通透，高细节",
    ),
}


if not torch.cuda.is_available():
    raise RuntimeError("No GPU is available. Please run this notebook inside a Radeon Cloud GPU instance.")

print("GPU:", torch.cuda.get_device_name(0))
print("Loading model. The first run may take several minutes if the model is not cached.")
model_dir = snapshot_download(MODEL_ID, cache_dir=CACHE_DIR)
pipe = ZImagePipeline.from_pretrained(
    model_dir,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
).to("cuda")
print("Model is ready.")


preset = widgets.Dropdown(
    options=list(PROMPT_PRESETS.keys()),
    value="高山湖泊旅行照",
    description="场景预设",
    layout=widgets.Layout(width="420px"),
)
prompt = widgets.Textarea(
    value=PROMPT_PRESETS["高山湖泊旅行照"][1],
    description="照片描述",
    layout=widgets.Layout(width="760px", height="96px"),
)
style = widgets.Dropdown(
    options=list(STYLE_PREFIX.keys()),
    value=PROMPT_PRESETS["高山湖泊旅行照"][0],
    description="摄影风格",
    layout=widgets.Layout(width="260px"),
)
size = widgets.Dropdown(
    options=["512x512", "768x768", "1024x1024"],
    value="768x768",
    description="图片尺寸",
    layout=widgets.Layout(width="220px"),
)
seed = widgets.IntText(value=42, description="随机种子", layout=widgets.Layout(width="220px"))

apply_preset_button = widgets.Button(description="套用预设", icon="check")
random_seed_button = widgets.Button(description="随机种子", icon="random")
generate_button = widgets.Button(description="生成图片", button_style="primary", icon="camera")
clear_button = widgets.Button(description="清空结果", icon="trash")

status = widgets.HTML(value="<b>状态：</b>模型已加载。请选择场景后点击生成图片。")
preview = widgets.Image(
    value=b"",
    format="png",
    layout=widgets.Layout(width="512px", max_width="100%", border="1px solid #ddd"),
)
preview_box = widgets.VBox([widgets.HTML("<b>生成预览</b>"), preview])
log_output = widgets.Output(layout=widgets.Layout(width="760px"))
history = widgets.HTML(value="<b>历史输出：</b>暂无")
generated_files = []


def set_busy(is_busy):
    generate_button.disabled = is_busy
    apply_preset_button.disabled = is_busy
    random_seed_button.disabled = is_busy
    clear_button.disabled = is_busy


def apply_preset(_=None):
    selected_style, selected_prompt = PROMPT_PRESETS[preset.value]
    style.value = selected_style
    prompt.value = selected_prompt
    status.value = f"<b>状态：</b>已套用预设：{preset.value}"


def randomize_seed(_=None):
    seed.value = random.randint(1, 2_147_483_647)
    status.value = f"<b>状态：</b>已生成随机种子：{seed.value}"


def update_history(path):
    generated_files.insert(0, str(path))
    shown = generated_files[:5]
    items = "".join(f"<li><code>{item}</code></li>" for item in shown)
    history.value = f"<b>历史输出：</b><ol>{items}</ol>"


def generate(_=None):
    height, width = map(int, size.value.split("x"))
    final_prompt = STYLE_PREFIX[style.value] + prompt.value.strip()
    generator = torch.Generator("cuda").manual_seed(int(seed.value))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"notebook_photo_{timestamp}_seed_{seed.value}.png"

    set_busy(True)
    status.value = "<b>状态：</b>正在生成，请等待..."
    with log_output:
        log_output.clear_output()
        print("Prompt:", final_prompt)
        print("Size:", size.value)
        print("Seed:", seed.value)
        start = time.time()
        try:
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

            OUTPUT_DIR.mkdir(exist_ok=True)
            image.save(output_path)
            preview.value = output_path.read_bytes()
            update_history(output_path.resolve())

            print(f"Saved to: {output_path.resolve()}")
            print(f"Generation time: {elapsed:.2f} seconds")
            status.value = f"<b>状态：</b>生成完成，用时 {elapsed:.2f} 秒。图片已显示并保存。"
        except Exception as exc:
            status.value = f"<b>状态：</b>生成失败：<code>{exc!r}</code>"
            raise
        finally:
            set_busy(False)


def clear_result(_=None):
    preview.value = b""
    log_output.clear_output()
    status.value = "<b>状态：</b>已清空当前预览，历史文件仍保留在 outputs 目录。"


preset.observe(lambda change: apply_preset() if change["name"] == "value" else None, names="value")
apply_preset_button.on_click(apply_preset)
random_seed_button.on_click(randomize_seed)
generate_button.on_click(generate)
clear_button.on_click(clear_result)

controls = widgets.VBox(
    [
        widgets.HTML("<h3>Radeon Cloud Z-Image 摄影生图控件</h3>"),
        widgets.HBox([preset, apply_preset_button, random_seed_button]),
        prompt,
        widgets.HBox([style, size, seed]),
        widgets.HBox([generate_button, clear_button]),
        status,
        preview_box,
        history,
        widgets.HTML("<b>运行日志</b>"),
        log_output,
    ]
)

display(controls)
