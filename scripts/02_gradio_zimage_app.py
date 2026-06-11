import time
import os
from pathlib import Path

import gradio as gr
import torch
from diffusers import ZImagePipeline
from modelscope import snapshot_download


MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "./models")
GRADIO_SHARE = os.environ.get("GRADIO_SHARE", "").lower() in {"1", "true", "yes", "y"}


if not torch.cuda.is_available():
    raise RuntimeError("No GPU is available. Please run this app inside a Radeon Cloud GPU notebook.")

print("GPU:", torch.cuda.get_device_name(0))
print("Loading model. The first run may take several minutes if the model is not cached.")
model_dir = snapshot_download(MODEL_ID, cache_dir=CACHE_DIR)
pipe = ZImagePipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
).to("cuda")


STYLE_PREFIX = {
    "人像写真": "真实摄影，人像写真，85mm镜头，浅景深，自然肤色，电影感光线，画面中不包含文字和水印，",
    "旅行风景照": "真实旅行摄影，广角镜头，自然风景，日出或日落光线，高细节，画面中不包含文字和水印，",
    "城市街拍": "真实城市街拍，35mm镜头，自然抓拍，街道环境光，电影色彩，画面中不包含文字和水印，",
    "户外运动照": "真实户外运动摄影，动态构图，清晰主体，自然光，高速快门质感，画面中不包含文字和水印，",
}


def generate(prompt, style, seed, size):
    height, width = map(int, size.split("x"))
    final_prompt = STYLE_PREFIX[style] + prompt
    generator = torch.Generator("cuda").manual_seed(int(seed))

    torch.cuda.synchronize()
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
    output_path = Path("outputs") / f"gradio_seed_{seed}.png"
    image.save(output_path)
    return image, f"生成耗时：{elapsed:.2f} 秒\n保存路径：{output_path.resolve()}"


with gr.Blocks(title="Radeon Cloud AI 生图应用") as demo:
    gr.Markdown("# Radeon Cloud AI 摄影生图应用")
    gr.Markdown("输入人物、风景或旅行场景描述，调用部署在 Radeon Cloud GPU 实例上的 Z-Image-Turbo 生成照片风格图片。")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="照片描述",
                value="一位年轻旅行者站在清晨的高山湖泊旁，远处雪山和薄雾，金色日出光线，自然表情",
                lines=4,
            )
            style = gr.Dropdown(list(STYLE_PREFIX.keys()), value="旅行风景照", label="摄影风格")
            size = gr.Dropdown(["512x512", "768x768", "1024x1024"], value="768x768", label="图片尺寸")
            seed = gr.Number(value=42, precision=0, label="随机种子")
            button = gr.Button("生成图片", variant="primary")
        with gr.Column():
            image = gr.Image(label="生成结果")
            info = gr.Textbox(label="运行信息")

    button.click(generate, inputs=[prompt, style, seed, size], outputs=[image, info])


demo.launch(server_name="0.0.0.0", server_port=7860, share=GRADIO_SHARE)
