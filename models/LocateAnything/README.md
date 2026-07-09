# LocateAnything-3B

本工程实现BM1684X/BM1688部署视觉定位大模型[LocateAnything-3B](https://huggingface.co/NVIDIA/LocateAnything-3B)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并将其部署到PCIE环境，或者SoC环境。

该模型支持图片中的目标定位（visual grounding），输入自然语言描述 + 图片，输出目标的 bounding box 坐标或 point 点坐标。

本文包括如何编译bmodel，和如何在BM1684X/BM1688环境运行bmodel。如何编译bmodel环节可以省去，直接用以下链接下载已编译好的bmodel（W4BF16，seq2048，max_input_length 1280，静态文本+动态ViT）：

``` shell
# BM1684X (PCIe / SoC, 1 dev)
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/locateanything-3b-autoround-w4a16_w4bf16_seq2048_bm1684x_1dev_static_20260709_154503.bmodel
# BM1688 (PCIe / SoC, 2 core)
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/locateanything-3b-autoround-w4a16_w4bf16_seq2048_bm1688_2core_static_20260709_162252.bmodel
```

## 编译LLM模型

此处介绍如何将LLM编译成bmodel。

#### 1. 从Huggingface下载模型

``` shell
git lfs install
git clone https://huggingface.co/groxaxo/LocateAnything-3B-AutoRound-W4A16
```

#### 2. 下载docker，启动容器

``` shell
docker pull sophgo/tpuc_dev:latest

docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
后文假定环境都在docker的`/workspace`目录。

#### 3. 下载`TPU-MLIR`代码并编译

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```

#### 4. 编译模型生成bmodel

``` shell
llm_convert.py -m /workspace/LocateAnything-3B-AutoRound-W4A16 \
  -q w4bf16 -s 2048 --max_input_length 1280 \
  -c bm1684x --max_pixels 896,896 --num_device 1 \
  -o /workspace/locateanything_bmodel
```

编译参数说明：
- `-q w4bf16`：文本LLM使用W4A16量化（AutoRound），ViT保持BF16
- `-s 2048`：最大序列长度（KV cache 总长，输入+输出）
- `--max_input_length 1280`：最大输入长度（图像 token + 文本 prompt），静态编译时 prefill padded 到此值
- `--max_pixels 896,896`：最大图片分辨率（对应64×64 patch grid，1024 image token）
- `-c bm1684x` / `-c bm1688`：目标芯片。bm1688 默认使用 2 core
- ViT 始终动态编译，支持任意尺寸图片输入；文本模型默认静态（prefill padded 到 `max_input_length`）。如需动态文本（prefill 按实际 token 数，小图更快），编译时加 `--dynamic`

## 编译与运行程序(python)

支持 PCIE 和 SoC 环境。SoC 环境下直接在设备上编译库文件（链接设备自带的 libsophon）。

### 1. 环境准备

需要 python3.10 环境。如果不满足，参考[此文档](https://github.com/sophgo/sophon-demo/blob/release/docs/FAQ.md#13-se7%E5%AE%89%E8%A3%85python310)安装。

``` shell
sudo apt-get update
sudo apt-get install pybind11-dev

pip3 install torch==2.6.0 torchvision==0.21.0 transformers==5.7.0 \
            pillow numpy lmdb opencv-python-headless
```

> 模型的 processor（trust_remote_code）顶层 import 了 `decord`/`lmdb`/`cv2`，transformers 5.7.0 的 `check_imports` 要求全部安装。`lmdb`、`opencv-python-headless` 在 aarch64 有 wheel；`decord` 在 aarch64（SoC）**无 PyPI wheel**：
> - PCIe（x86）：`pip install decord` 即可。
> - SoC（aarch64）仅图片部署时，可创建 stub 模块让 import 通过（video 路径不会执行）：
>   ``` shell
>   SP=$(python3.10 -c "import site;print(site.getusersitepackages())")
>   mkdir -p "$SP" && cat > "$SP/decord.py" <<'EOF'
>   class VideoReader:
>       def __init__(self,*a,**k):
>           raise ImportError("decord not installed on aarch64; video unsupported")
>   EOF
>   ```
>   需要视频则从源码编译 decord（依赖 ffmpeg-dev）。

### 2. 编译库文件

编译C++库文件，生成`chat.cpython*.so`：

``` shell
cd python_demo
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

> SoC 上若 cmake 选中了非 3.10 的默认 python（如 python3.8），显式指定：
> `cmake -DPython_EXECUTABLE=$(which python3.10) ..`

### 3. 运行demo

``` shell
python3 pipeline.py -m /path/to/model.bmodel -c ../config
```

交互模式下输入描述文本和图片路径即可获得 bounding box 坐标。也可用 `-p` 单次推理模式（见下方示例）。

### 运行示例

支持以下任务（prompt 模板与源模型一致）：

**单目标检测 / 短语定位**

``` shell
python3 pipeline.py -m model.bmodel -c ../config -p "detect bed" --media_path test.jpg
# token 输出：<ref>bed</ref><box><0><585><627><998></box>
# 坐标为归一化值 [0,1000]，对应 (x1,y1,x2,y2)
# 解析输出（像素坐标）：[bed] box (0,283)-(401,482)
```

**多目标检测 / 密集检测**

对一个 `<ref>` 可输出多个 `<box>`，支持多类别（用 `</c>` 分隔）：

``` shell
python3 pipeline.py -m model.bmodel -c ../config \
  -p "Locate all the instances that matches the following description: bed</c>window</c>pillow" \
  --media_path test.jpg
# 解析输出（保留 ref 关联，bed 1 框、window 1 框、pillow 16 框）：
#   [bed] box (0,283)-(401,482)
#   [window] box (259,36)-(394,226)
#   [pillow] box (0,283)-(57,317)
#   ...（共 16 个 pillow 框）
```

**点定位（Pointing）**

输出 2 个坐标 (x, y)：

``` shell
python3 pipeline.py -m model.bmodel -c ../config -p "Point to: bed" --media_path test.jpg
# token 输出：<ref>bed</ref><box><333><811></box>
# 解析输出（像素坐标）：[bed] point (213,392)
```

> pipeline 会自动将归一化坐标 [0,1000] 解析为像素坐标（`parse_boxes`/`parse_points`/`parse_result`），并按 `<ref>` 分组打印。token 原文也会实时打印。

## 技术说明

- **量化方案**：文本 LLM（Qwen2.5-3B）使用 AutoRound W4A16 对称量化；ViT（MoonViT-SO-400M）和 MLP1 投影器保持 BF16
- **推理模式**：当前仅支持 slow 模式（纯自回归），MTP 多 token 预测待后续实现
- **批量推理**：暂不支持批量推理（源模型提供 `batch_infer.py` + `la_flash` 后端），当前仅单张图片串行推理
- **ViT 动态编译**：支持任意图片尺寸，pipeline 动态计算 pos_emb（bicubic 插值）、2D RoPE cos/sin、merger_index 并传入 ViT bmodel

### 图片 token 数计算

图片先按 14×14 切 patch，再 2×2 空间合并（merge），因此每 28×28 像素块对应 1 个 image token。processor 会将图片 resize 到满足 `max_pixels` 与每维 28 像素对齐（`merge_kernel_size × patch_size = 2 × 14`）的尺寸，再计算 grid。

- 公式：`image_tokens = (grid_h × grid_w) / 4`，其中 `grid_h = resized_H / 14`、`grid_w = resized_W / 14`
- 等价地：`image_tokens ≈ 图片像素数 / 784`
- 另有 2 个包装 token（`<img>`、`</img>`）将 image token 包住
- `max_pixels 896,896` 时上限为 **1024 个 image token**

| 图片尺寸 | grid (h×w) | image token 数 |
|---------|-----------|---------------|
| 640×483 | 36×46 | 414 |
| 896×896 | 64×64 | 1024 |

序列长度 `-s 2048`（KV cache 总长）需容纳：image token + 文本 prompt + 生成的回答。`--max_input_length 1280` 限制输入上限（图像 1024 + 文本），输出预算 = 2048 − 1280 = 768 token；大图会占用更多输入预算，留给生成回答的空间更少。

