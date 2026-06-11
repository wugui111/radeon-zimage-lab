import time
import os
from pathlib import Path

import gradio as gr
import torch
from diffusers import ZImagePipeline
from modelscope import snapshot_download


MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "./models")


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
    "校园科技海报": "校园科技海报风格，蓝白配色，现代排版，中文标题清晰，",
    "电商商品图": "电商商品宣传图风格，干净背景，主体突出，商业摄影质感，",
    "课程封面": "在线课程封面风格，信息层级清楚，适合作为教学资源封面，",
    "国潮插画": "国潮插画风格，中国传统纹样，现代视觉设计，",
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
    gr.Markdown("# Radeon Cloud AI 生图应用")
    gr.Markdown("输入文字描述，调用部署在 Radeon Cloud GPU 实例上的 Z-Image-Turbo 生成图片。")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="图片描述",
                value="为软件技术专业实训周设计一张中文宣传海报，主题是云计算与人工智能应用开发",
                lines=4,
            )
            style = gr.Dropdown(list(STYLE_PREFIX.keys()), value="校园科技海报", label="风格")
            size = gr.Dropdown(["512x512", "768x768", "1024x1024"], value="768x768", label="图片尺寸")
            seed = gr.Number(value=42, precision=0, label="随机种子")
            button = gr.Button("生成图片", variant="primary")
        with gr.Column():
            image = gr.Image(label="生成结果")
            info = gr.Textbox(label="运行信息")

    button.click(generate, inputs=[prompt, style, seed, size], outputs=[image, info])


demo.launch(server_name="0.0.0.0", server_port=7860)
