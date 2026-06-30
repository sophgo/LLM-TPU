# MiniCPM-V-4.6

本工程实现BM1684X/BM1688部署多模态大模型[MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并将其部署到PCIE环境，或者SoC环境。

该模型支持图片和视频的识别，有python版本的demo。

本文包括如何编译bmodel，和如何在BM1684X/BM1688环境运行bmodel。如何编译bmodel环节可以省去，直接用以下链接下载：

``` shell
# =============== 1684x =====================
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm-v-4.6_bf16_seq2048_bm1684x_1dev_dynamic_20260630_105643.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm-v-4.6_bf16_seq2048_bm1688_2core_dynamic_20260630_155028.bmodel
```

## 模型架构

MiniCPM-V-4.6 参数量 1.3B，由视觉编码器和文本模型两部分组成：

- **文本模型**：与 Qwen3.5 相同架构
- **视觉编码器**：SigLIP ViT + Merger 降采样，支持两种模式：
  - `16x`（默认）：两级 2×2 合并，共 16 倍降采样，token 数少，推理快
  - `4x`：一级 2×2 合并，4 倍降采样，保留更多视觉细节

## 编译bmodel

此处介绍如何将模型编译成bmodel。

#### 1. 下载模型

``` shell
git clone https://huggingface.co/openbmb/MiniCPM-V-4.6
```

#### 2. 下载docker，启动容器

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
后文假定环境都在docker的`/workspace`目录。

#### 3. 下载`TPU-MLIR`代码并编译

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  #激活环境变量
./build.sh #编译mlir
```

#### 4. 编译模型生成bmodel

``` shell
# 这里max_input_length指定最大输入长度，如果不指定则为-s指定的长度
llm_convert.py -m /workspace/MiniCPM-V-4.6 -s 2048 --max_input_length 1024 -q bf16 -c bm1684x -o minicpm_v4_6 --max_pixels 448,448
```
编译完成后，在指定目录生成`minicpm-v-4.6-xxx.bmodel`和`config`。

## 编译与运行程序(python)

### 1. 环境准备

需要 python3.10 环境。如果不满足，参考[此文档](https://github.com/sophgo/sophon-demo/blob/release/docs/FAQ.md#13-se7%E5%AE%89%E8%A3%85python310)安装。

``` shell
sudo apt-get update
sudo apt-get install pybind11-dev

pip3 install torch==2.6.0 torchvision==0.21.0 transformers==5.7.0
```

### 2. 编译库文件

编译C++库文件，生成`chat.cpython*.so`：

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

### 3. 运行

``` shell
# 交互模式
python3 pipeline.py -m minicpm-v-4.6.bmodel -c ../config

# 单次推理模式（图片）
python3 pipeline.py -m minicpm-v-4.6.bmodel -c ../config \
    --prompt "描述这张图片" --media_path test.jpg

# 单次推理模式（视频）
python3 pipeline.py -m minicpm-v-4.6.bmodel -c ../config \
    --prompt "描述视频中发生了什么" --media_path test.mp4
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-m, --model_path` | 必填 | bmodel 文件路径 |
| `-c, --config_path` | `../config` | processor 配置文件目录 |
| `-d, --devid` | `0` | 设备 ID |
| `--downsample_mode` | `16x` | ViT 降采样模式，可选 `4x` / `16x` |
| `--max_slice_nums` | None | 图片最大切片数。不指定时图片默认36，视频默认1；指定后统一使用 |
| `--max_num_frames` | `16` | 视频最大采样帧数 |
| `-p, --prompt` | None | 指定后进入单次推理模式 |
| `-t, --prompt_file` | None | 从文件加载 prompt |
| `--media_path` | 空 | 图片或视频路径，配合 `--prompt` 使用 |

## 进阶应用

参考[Qwen3.5 README](../Qwen3_5/README.md)。

## 常见问题

#### 一张图片占多少 Token ?

计算公式：$token数 = \frac{h_{patches} \times w_{patches}}{merge\_size^2}$

其中 $h_{patches} = \frac{height}{14}$，$w_{patches} = \frac{width}{14}$。

以 448×448 图片为例：
- patches = 32 × 32 = 1024
- 16x 模式：1024 / 16 = **64 tokens**
- 4x 模式：1024 / 4 = **256 tokens**

高分辨率图片会被切片处理，`max_slice_nums` 控制最大切片数。

#### 视频占多少 Token ?

视频每帧独立处理，`max_slice_nums=1`（不切片）。

以 16x 模式、448×448 每帧为例，每帧 64 tokens。

16 帧视频：$64 × 16 = 1024$ tokens

#### 视频最多支持多少帧 ?

| 降采样模式 | 每帧 Token | 最大帧数 (SEQLEN=2048) |
|-----------|-----------|----------------------|
| 16x | ~64 | ~25 |
| 4x | ~256 | ~6 |

实际可用帧数还需减去 prompt 和输出占用的 tokens。
