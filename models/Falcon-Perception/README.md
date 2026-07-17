# Falcon-Perception

本工程实现 BM1684X 部署视觉分割大模型 [Falcon-Perception](https://github.com/tiiuae/Falcon-Perception)（HF: `tiiuae/falcon-perception`）。通过 [TPU-MLIR](https://github.com/sophgo/tpu-mlir) 编译器将模型转换成 bmodel，并部署到 BM1684X PCIe / SoC 环境。

该模型支持图片中的目标分割（referring segmentation）：输入自然语言描述 + 图片，输出目标的归一化 bounding box（`xy` 中心点、`hw` 宽高）+ 二值分割 mask。本文包括如何编译 bmodel，以及如何在 BM1684X（PCIe / SoC）环境运行。

可直接用以下链接下载已编译好的 bmodel（F32，seq512，max_input_length 384，静态，256×256 图片，PCIe / SoC 通用）：

``` shell
pip install dfss
# BM1684X (PCIe / SoC, 1 dev) — 约 2.5GB
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/falcon-perception_f32_seq512_bm1684x_1dev_static_20260716_185805.bmodel
```

## 模型架构

Falcon-Perception 是一个基于 Falcon 主干（28 层、dim 1024、16 头、8 KV 头、GQA）的 referring-segmentation 模型，F32 全精度部署：

- **主干 LLM**：Falcon-style decoder（unweighted RMSNorm、squared-ReLU-gate FFN、2D golden RoPE、attention sink）。prefill + decode（block_cache）两段编译。
- **视觉上采样（anyup）**：输入图片 + 主干 image-token 隐状态 + window mask，经 Fourier encoder / key encoder / LFU / resblock 聚合，输出 256×256×256 hr_features。
- **头**：coord / size / seg / mask 四个头 + coord_encoder / size_encoder 两个 Fourier 回灌编码器。decode 循环按 token id 分派：coord_token(240)→coord 头出 (x,y)，size_token→size 头出 (h,w)，seg_token→seg 头出 seg_vec → mask 头出 256×256 mask。coord/size 的 Fourier 编码在下一步 forward_next 前回灌覆盖对应 token 的 embedding。

## 编译 bmodel

#### 1. 从 HuggingFace 下载模型

``` shell
git lfs install
git clone https://www.modelscope.cn/tiiuae/Falcon-Perception.git
```

#### 2. 下载 docker，启动容器

``` shell
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

#### 3. 下载 `TPU-MLIR` 代码并编译

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir 
source ./envsetup.sh 
./build.sh
```

#### 4. 编译模型生成 bmodel

``` shell
llm_convert.py -m /workspace/falcon-perception \
  -s 512 --max_input_length 384 -q f32 -c bm1684x --max_pixels 256,256 \
  -o /workspace/falcon-perception_bmodel
```

编译参数说明：
- `-q f32`：F32 全精度（不量化）。
- `-s 512`：KV cache 总长（SEQLEN）。
- `--max_input_length 384`：prefill 输入上限。
- `--max_pixels 256,256`：图片最大像素 256×256。
- `-c bm1684x`：目标芯片。
- 静态编译（固定 shape）。超长 query 或 >~16 次检测的场景需用户自行调大 `-s`/`--max_input_length` 重编。

产物 `falcon-perception_f32_seq512_bm1684x_1dev_static_<ts>.bmodel`（约 2.5GB，64 net）。

## 运行推理（Python）

### 1. 环境准备

需要 python3.10 + pybind11。若设备未装 python3.10，参考 [此文档](https://github.com/sophgo/sophon-demo/blob/release/docs/FAQ.md#13-se7%E5%AE%89%E8%A3%85python310) 安装。

``` shell
sudo apt-get update && sudo apt-get install pybind11-dev
pip3 install torch==2.6.0 transformers==5.7.0 pillow numpy einops requests
```

`config/` 目录需包含 HF 模型的 `config.json`、自定义 processor（`configuration_falcon_perception.py`、`processing_falcon_perception.py`）、tokenizer，以及额外提取的 `falcon_extra_weights.npz`（img_projector + golden rope freqs，约 3MB，由 Converter 生成）。anyup 的 window attention mask 是静态几何常量，已烘进 bmodel，不在 npz 里。

### 2. 编译库文件

``` shell
cd python_demo
mkdir build && cd build && cmake .. && make -j4 && cp *cpython* .. && cd ..
```

> SoC（aarch64）原生编译（非交叉）。若 cmake 选中非 3.10 的默认 python，显式指定：`cmake -DPython_EXECUTABLE=$(which python3.10) ..`

### 3. 运行 demo

**单次推理**：

``` shell
python3 pipeline.py -m falcon-perception.bmodel -c ../config \
  -q "the bed" --media_path test.jpg
```

**交互模式**（不加 `-q`）：依次输入 Query 和 Image Path，`exit` 退出。

### 运行示例

``` shell
python3 pipeline.py -m ../falcon-perception.bmodel -c ../config -q "the bed" --media_path test.jpg
# [info] tokens=209  img_tokens=192
# Answer:
# <|presence|>
# [coord] x=0.3128 y=0.7928
# [size] h=0.4089 w=0.6266
# [seg] mask_pos=8525
#
# [emission] coord=1 size=1 seg=1  [detections] 1 (NMS kept 1/1)
#   #0 xy=(0.313,0.793) hw=(0.409,0.627) mask_px=53804/309120
#   visualization: ./test_the_bed_vis.jpg
# FTL: 1.33 s   decode: 1.47 s   tokens: 4   TPS: 2.70
```

pipeline 在原 256×256 mask logits 上做完整后处理（对齐 HF `_postprocess_aux`）：

- `xy`：目标框中心点归一化坐标 [0,1]（→ 像素 `x*W, y*H`）。
- `hw`：目标框宽高归一化 [0,1]（半宽 `w*W/2`、半高 `h*H/2`）。
- `mask_px`：原图分辨率二值 mask 的前景像素数（`/` 后为总像素 H×W）。
- token 流（与 HF 一致）：`<|presence|>` → `<|coord|>` → `<|size|>` → `<|seg|>` → `<|end_of_query|>`。每次检测 = coord + size + seg 三个任务 token；coord/size token 各编码一个 2D 坐标（coord 头出 2×1024 bin 分布、argmax 取 x/y，size 头同理取 h/w），不是逐坐标一个 token；mask 由 seg token 的隐藏态经 mask 头一次产生、不占 decode token。

## CLI 参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-m` / `--model_path` | （必填） | bmodel 路径 |
| `-c` / `--config_path` | `../config` | 模型 config 目录（config.json + 自定义 processor + falcon_extra_weights.npz） |
| `-d` / `--devid` | `0` | TPU 设备号 |
| `-q` / `--query` | `None` | 设定后单次推理并退出；不设则进交互模式 |
| `--media_path` | `""` | 图片路径（单次推理模式） |

## 技术说明

- **量化方案**：F32 全精度，不量化。主干、anyup、各头均为 F32。
- **采样**：固定 greedy（temperature=0），不支持 top-k / 采样 / seed（HF `generate` 有但本 demo 未暴露）。
- **Batch**：不支持 batch 推理，单图单 query 串行。
- **上下文长度**：seq512/max_input384 覆盖 256×256 下大部分场景（query ≤ ~700 字符、≤ ~18 次检测）。极端场景由用户重编调大。
- **历史记录**：不支持多轮历史（单图单 query）。

## 常见问题

- **支持的分辨率**：最大 256×256（`--max_pixels 256,256`）。改分辨率需重编bmodel。
- **图像 token 数**：图像按保持长宽比 resize 到 ≤256×256，每边 round 到 16 的倍数（patch_size=16、merge=1），`image_tokens = (H_res/16) × (W_res/16)`。例：640×483 → 256×192 → 16×12 = 192 tokens；方图 256×256 则上限 256 tokens。再加模板 token + query 即为 prefill 长度。
- **显存**：约 3.3GB（DevMem 3363/14678 MB on BM1684X）。
