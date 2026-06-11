# Radeon Cloud 部署 Z-Image-Turbo 生图模型实验指导

适用对象：软件技术、云计算技术相关专业学生
实验平台：Radeon Cloud，https://radeon.anruicloud.com/
实验模型：Tongyi-MAI/Z-Image-Turbo
建议课时：基础实验 30-45 分钟；网页应用 1 学时；LoRA 微调 2-4 学时或课后项目

## 一、实验目标

完成本实验后，学生应能完成以下任务：

1. 登录 Radeon Cloud 并启动一个 GPU Notebook/Workspace。
2. 在云端实例中安装依赖并部署 Z-Image-Turbo。
3. 使用 Python 脚本输入提示词并生成图片。
4. 将生图能力封装成一个简单网页应用。
5. 理解“通用下载部署”和“教师预缓存镜像部署”两种课堂组织方式。
6. 了解使用少量图片进行 LoRA 微调的基本流程。

本实验：云端 GPU 资源如何支撑一个可运行的 AI 生图应用。

## 二、实验材料

本目录已经准备好以下文件：

```text
/Users/keshiwei/C/RadeonCloud
├── RadeonCloud_ZImage_Turbo_实验指导.md
├── scripts
│   ├── 00_check_gpu.py
│   ├── 01_generate_zimage.py
│   ├── 02_gradio_zimage_app.py
│   ├── 03_prepare_lora_dataset.py
│   ├── 04_train_lora_reference.sh
│   ├── 05_warmup_model_cache.py
│   └── install_deps.sh
├── docker
│   ├── Dockerfile.zimage
│   ├── build_and_push_zimage.sh
│   └── README.md
└── assets
    └── 若干实验截图
```

教师课前建议将 `scripts` 目录上传到一个 GitHub 仓库，例如：

```text
https://github.com/<teacher-name>/radeon-zimage-lab
```

如果不使用 GitHub，也可以在 Jupyter Notebook 中手动上传这些脚本文件。

## 三、平台登录与入口

### 步骤 1：打开 Radeon Cloud

在 Chrome 或 Edge 中打开：

```text
https://radeon.anruicloud.com/
```

### 步骤 2：登录平台

点击右上角登录入口，使用教师提供的账号方式登录。登录成功后，右上角会显示用户名或头像。

![](assets/20260611_083317_image.png)

### 步骤 3：认识页面上的关键按钮

Gallery 页面常用按钮如下：

- `Create Template`：教师创建课程模板时使用。
- `Preview`：预览已有 Notebook 模板。
- `Launch`：启动 GPU Notebook/Workspace。
- 顶部 `Gallery`：模板库。
- 顶部 `Space`：查看自己的运行空间或实例。

课堂上学生通常只需要点击某个模板卡片下方的 `Launch`。

## 四、基础实验：通用下载部署方案

本方案适合第一次验证流程。缺点是首次运行会下载 Z-Image-Turbo 模型，模型文件较大，课堂上可能等待较久。

### 步骤 1：启动 Blank Notebook

在 Gallery 页面找到 `Blank Notebook` 模板。

如果当前页面第一屏没有看到它，可以：

1. 在搜索框输入 `Blank Notebook`。
2. 或向下滚动 Gallery。
3. 找到卡片后点击 `Launch`。

注意：点击 `Launch` 会申请 GPU 云端资源。课堂中建议由教师统一安排启动时间，避免学生同时反复启动实例。

### 步骤 2：等待 Notebook 就绪

![](assets/20260611_083208_image.png)

点击 `Launch` 后，平台会显示启动状态。等待状态变为 ready 或出现打开 Notebook 的入口。

如果页面提示只能有一个活动实例，请进入 `Space` 查看已有实例，复用或关闭旧实例。

### 步骤 3：打开 Notebook 终端

![](assets/20260611_083426_image.png)

进入 Notebook 后，打开 Terminal。通常路径是：

```text
Notebook 页面 -> Launcher -> Terminal
```

![](assets/20260611_083645_image.png)

或者在左侧文件区中新建 Notebook，再通过单元格运行命令。

### 步骤 4：获取实验脚本

如果教师已经准备 GitHub 仓库，在终端执行：

```bash
apt-get update
apt-get install -y ca-certificates git
update-ca-certificates
git config --global http.sslCAInfo /etc/ssl/certs/ca-certificates.crt

git clone https://github.com/wugui111/radeon-zimage-lab.git
cd radeon-zimage-lab
```

如果教师使用其他仓库，请把上面的仓库地址替换成实际地址。不要优先使用关闭 SSL 校验的方式克隆仓库。

### 步骤 5：检查 GPU

在云端终端运行：

```bash
python scripts/00_check_gpu.py
```

预期输出中应包含：

```text
CUDA/ROCm interface available: True
GPU name: ...
GPU memory: ...
```

说明：在 ROCm/PyTorch 环境中，AMD GPU 通常仍通过 `cuda` 这个接口名暴露给 Python，看到 `cuda` 是正常现象。

### 步骤 6：安装依赖

本实验建议使用独立虚拟环境，避免影响平台基础镜像中已有的 vLLM、transformers 等包。运行：

```bash
bash scripts/install_deps.sh
```

该脚本会自动完成三件事：

1. 在容器内安装 `ca-certificates` 和 `git`，修复 GitHub 证书校验失败问题。
2. 创建 `.venv` 虚拟环境，并使用 `--system-site-packages` 继承平台已有的 ROCm/PyTorch。
3. 在 `.venv` 中安装本实验需要的依赖，并将 `transformers` 固定为 `<5`，避免破坏基础镜像中的 vLLM 依赖。

安装完成后，先激活虚拟环境：

```bash
source .venv/bin/activate
```

然后再次检查 GPU：

```bash
python scripts/00_check_gpu.py
```

脚本会安装：

- `modelscope`
- `diffusers`
- `transformers>=4.56,<5`
- `accelerate`
- `gradio`
- `safetensors`
- `pillow`

如果安装过程中网络较慢，请耐心等待。教师也可以提前使用后文的“懒人镜像方案”避免课堂等待。

### 步骤 7：运行命令行生图脚本

运行：

```bash
source .venv/bin/activate
python scripts/01_generate_zimage.py
```

默认提示词为：

```text
为软件技术专业实训周设计一张中文宣传海报，主题是云计算与人工智能应用开发，蓝白科技风，清晰中文标题，现代设计，高质量
```

生成完成后，图片会保存到：

```text
outputs/zimage_result.png
```

也可以自定义提示词：

```bash
python scripts/01_generate_zimage.py \
  --prompt "为学校社团招新活动设计一张中文海报，青春活力，明亮色彩，清晰中文标题" \
  --height 768 \
  --width 768 \
  --seed 123 \
  --output outputs/club_poster.png
```

### 步骤 8：查看生成结果

在 Jupyter 文件区打开 `outputs` 目录，双击生成的 `.png` 图片查看。

学生提交截图时至少包含：

1. GPU 检测输出。
2. 生图脚本运行输出。
3. 生成图片。

## 五、进阶实验：部署网页生图应用

基础脚本跑通后，可以进一步做一个网页应用。

### 步骤 1：启动 Gradio 应用

在云端终端运行：

```bash
source .venv/bin/activate
python scripts/02_gradio_zimage_app.py
```

启动成功后，终端会显示类似：

```text
Running on local URL: http://0.0.0.0:7860
```

### 步骤 2：打开网页

根据 Radeon Cloud/Jupyter 的端口代理方式打开 `7860` 端口。

常见方式有三种：

1. Notebook 页面自动显示 Gradio 链接，直接点击。
2. Jupyter 提供端口转发入口，选择 `7860`。
3. 如果平台没有暴露端口，请在 Notebook 单元格中运行 Gradio，直接在输出区域使用。

### 步骤 3：使用网页生成图片

网页包含以下控件：

- `图片描述`：输入想生成的内容。
- `风格`：选择校园科技海报、电商商品图、课程封面、国潮插画。
- `图片尺寸`：建议课堂使用 `768x768` 或 `512x512`。
- `随机种子`：相同提示词和种子通常生成相近结果。
- `生成图片`：点击后开始推理。

### 步骤 4：学生小任务

让学生完成以下任务之一：

1. 生成一张“云计算课程海报”。
2. 生成一张“软件技术专业实训周海报”。
3. 生成一张“学校社团招新海报”。
4. 生成一张“电商商品宣传图”。

提交内容：

- 网页界面截图。
- 生成图片截图。
- 所使用的提示词。

## 六、懒人部署方案：教师预缓存镜像，学生免下载

### 适用场景

如果全班学生都在课堂上首次下载 Z-Image-Turbo，会出现以下问题：

- 模型文件较大，下载时间不可控。
- 多人同时下载会占用带宽。
- 学生容易长时间等待，课堂体验不好。

因此建议教师课前准备一个“已安装依赖、已缓存模型”的容器镜像。

### 平台可行性判断

Radeon Cloud 的 Gallery 页面提供 `Create Template` 按钮。页面源码中可以看到创建模板表单包含以下字段：

- `Title`
- `Description`
- `Category`
- `Tags`
- `Container Image`
- `GitHub Repo URL`
- `Branch`
- `Notebook Path`
- `Cover URL`

这说明平台支持基于已有容器镜像创建模板。但它看起来不是在网页中直接构建镜像，而是选择一个已经存在或已经被平台允许的容器镜像。

![Create Template 入口位置](assets/20260611_083317_image.png)

### 教师课前镜像构建

本目录提供了参考 Dockerfile：

```text
docker/Dockerfile.zimage
```

它做了三件事：

1. 基于 Radeon Cloud 的 ROCm/PyTorch 基础镜像。
2. 安装 Z-Image-Turbo 推理所需 Python 依赖。
3. 预下载 `Tongyi-MAI/Z-Image-Turbo` 到 `/opt/models`。

构建并推送镜像：

```bash
cd /Users/keshiwei/C/RadeonCloud

REGISTRY_IMAGE=registry.example.com/your-namespace/radeon-zimage-turbo:rocm7.2 \
  bash docker/build_and_push_zimage.sh
```

请把 `REGISTRY_IMAGE` 替换成 Radeon Cloud 可访问的镜像仓库地址。

### 在 Radeon Cloud 中创建懒人模板

教师登录 Radeon Cloud 后：

1. 进入 `Gallery`。
2. 点击 `Create Template`。
3. 填写 `Title`，例如：

```text
Z-Image-Turbo 生图实验
```

4. 填写 `Description`，例如：

```text
已缓存 Z-Image-Turbo 模型，学生可直接运行脚本生成图片
```

5. `Category` 选择或填写：

```text
Courses
```

6. `Tags` 填写：

```text
z-image, image-generation, rocm
```

7. `Container Image` 选择教师提前推送的镜像：

```text
registry.example.com/your-namespace/radeon-zimage-turbo:rocm7.2
```

8. `GitHub Repo URL` 填写教师脚本仓库地址：

```text
https://github.com/<teacher-name>/radeon-zimage-lab.git
```

9. `Branch` 填写：

```text
main
```

10. `Notebook Path` 可填写教师准备好的 Notebook，例如：

```text
notebooks/zimage_lab.ipynb
```

如果只使用终端脚本，可以留空。

11. 点击 `Create Template`。

注意：如果 `Container Image` 下拉框中看不到教师镜像，说明平台可能需要管理员先加入镜像白名单或镜像列表。此时请联系平台管理员创建模板或开放该镜像。

### 学生使用懒人模板

学生只需要：

1. 登录 Radeon Cloud。
2. 在 Gallery 搜索 `Z-Image-Turbo 生图实验`。
3. 点击模板卡片下方的 `Launch`。
4. 进入 Notebook 后直接运行：

```bash
python scripts/00_check_gpu.py
python scripts/01_generate_zimage.py
```

因为镜像中已经预缓存模型，首次运行不需要再下载几十 GB 模型，课堂体验会明显好很多。

## 七、项目拓展：LoRA 微调自己的生图风格

### 重要说明

LoRA 微调不建议放进 30 分钟基础课堂。它更适合作为 2-4 学时项目或课后拓展。

推荐项目题目：

- 校园建筑风格海报生成模型。
- 专业课程封面生成模型。
- 电商商品宣传图生成模型。
- 班级 IP 形象生成模型。

### 步骤 1：准备图片数据

准备 10-30 张图片，放入一个目录，例如：

```text
my_images/
├── 001.jpg
├── 002.jpg
└── ...
```

图片建议：

- 内容风格一致。
- 分辨率不要过低。
- 不使用含隐私、肖像权不明或版权不明的图片。

### 步骤 2：生成数据集 metadata

运行：

```bash
python scripts/03_prepare_lora_dataset.py \
  --image-dir ./my_images \
  --caption "校园建筑风格，蓝白科技风，现代中文海报设计" \
  --output-dir ./lora_dataset
```

输出目录结构：

```text
lora_dataset/
├── images/
│   ├── image_001.jpg
│   └── ...
└── metadata.csv
```

### 步骤 3：安装 DiffSynth-Studio

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

### 步骤 4：参考官方 Z-Image-Turbo LoRA 命令

官方示例命令如下：

```bash
modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset \
  --include "z_image/Z-Image-Turbo/*" \
  --local_dir ./data/diffsynth_example_dataset

accelerate launch examples/z_image/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/z_image/Z-Image-Turbo \
  --dataset_metadata_path data/diffsynth_example_dataset/z_image/Z-Image-Turbo/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Tongyi-MAI/Z-Image-Turbo:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Z-Image-Turbo_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,w1,w2,w3" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8
```

如果使用自己的数据集，需要把参数替换为：

```bash
--dataset_base_path ../lora_dataset
--dataset_metadata_path ../lora_dataset/metadata.csv
```

### 微调提示

Z-Image-Turbo 是蒸馏加速模型，直接训练可能影响它的快速生成能力。课堂项目中建议只把它作为体验 LoRA 微调流程使用，不要承诺一定得到稳定高质量模型。

## 八、常见问题

### 1. 为什么代码里写的是 `cuda`，但平台是 AMD GPU？

PyTorch ROCm 环境通常复用 `torch.cuda` 接口，因此 AMD GPU 也会通过 `cuda` 接口被识别。这是正常现象。

### 2. 第一次运行为什么很慢？

首次运行需要安装依赖并下载模型。建议教师课前使用懒人镜像方案，把依赖和模型缓存进镜像。

### 3. 安装依赖时报 GitHub 证书错误怎么办？

如果看到类似错误：

```text
server certificate verification failed. CAfile: none CRLfile: none
```

说明容器里缺少 CA 根证书。新版 `scripts/install_deps.sh` 已经自动处理。也可以手动运行：

```bash
apt-get update
apt-get install -y ca-certificates git
update-ca-certificates
git config --global http.sslCAInfo /etc/ssl/certs/ca-certificates.crt
```

然后重新执行：

```bash
bash scripts/install_deps.sh
source .venv/bin/activate
```

### 4. 安装依赖时提示 `transformers` 或 `starlette` 版本冲突怎么办？

这是因为基础镜像中可能预装了 vLLM 等工具，它们对依赖版本有要求。本实验只为 Z-Image 使用，推荐使用 `.venv` 虚拟环境：

```bash
bash scripts/install_deps.sh
source .venv/bin/activate
```

如果已经在全局环境里装过最新版 `transformers`，可以在当前虚拟环境中重新固定版本：

```bash
pip install -U "transformers>=4.56,<5" "starlette>=0.30,<1"
```

课堂中不要执行全局关闭证书校验命令，例如 `git config --global http.sslVerify false`，除非是临时容器且用完立即销毁。

### 5. 显存不够怎么办？

可以先降低图片尺寸：

```bash
source .venv/bin/activate
python scripts/01_generate_zimage.py --height 512 --width 512
```

也可以在后续版本中加入 CPU offload 或使用 DiffSynth-Studio 的低显存推理方案。

### 6. Gradio 网页打不开怎么办？

优先检查：

1. 终端是否显示 `Running on local URL`。
2. 平台是否提供端口代理。
3. 是否需要在 Notebook 输出区直接使用 Gradio。

如果平台没有开放端口代理，可以把网页应用作为教师演示，学生只完成 Python 脚本版。

### 7. 学生需要提交什么？

基础实验提交：

- GPU 检查截图。
- 运行 `01_generate_zimage.py` 的终端截图。
- 生成图片文件。
- 100-200 字实验总结。

进阶实验提交：

- Gradio 网页截图。
- 输入提示词。
- 生成图片。
- 对网页应用结构的简要说明。

## 九、课堂建议

30 分钟课堂建议只做基础实验：

1. 登录平台。
2. 启动教师预置模板。
3. 检查 GPU。
4. 运行生图脚本。
5. 修改提示词再生成一次。
6. 提交截图。

网页应用适合第二节课；LoRA 微调适合大作业。
