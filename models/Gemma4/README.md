# Gemma4

Gemma4 是 Google 推出的多模态大语言模型，支持文本、图像、视频和音频输入。

- Gemma4-E2B-it: 2.3B 有效参数（含嵌入层共 5.1B）
- Gemma4-E4B-it: 4.5B 有效参数（含嵌入层共 8B）

以下是编译好的模型，可以直接下载：
```bash
pip3 install dfss
# E2B BM1684X 2k上下文
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/gemma-4-e2b-it-awq-4bit_w4f16_seq2048_bm1684x_1dev_static_20260618_152146.bmodel
# E2B BM1684X 4k上下文
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/gemma-4-e2b-it-awq-4bit_w4f16_seq4096_bm1684x_1dev_static_20260618_184549.bmodel
# E2B BM1688 2k上下文
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/gemma-4-e2b-it-awq-4bit_w4f16_seq2048_bm1688_2core_static_20260618_143212.bmodel
# E2B BM1688 4k上下文
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/gemma-4-e2b-it-awq-4bit_w4f16_seq4096_bm1688_2core_static_20260617_212224.bmodel
# E2B 嵌入层权重，BM1684X 和 BM1688 都需要
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/weight/per_layer_token_embd_e2b.bin

# E4B BM1684X
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/gemma-4-e4b-it-awq-4bit_w4f16_seq4096_bm1684x_1dev_static_20260521_154219.bmodel
```

## 1. 编译 bmodel

### 1.1 下载源模型

比较大，会花费较长时间。 请下载 awq 或者 gptq 量化的版本，或者自行量化
```bash
# E2B
git clone https://huggingface.co/Chunity/gemma-4-E2B-it-AWQ-4bit
# E4B
git clone https://huggingface.co/Chunity/gemma-4-E4B-it-AWQ-4bit
```

也可自行编译模型。

### 1.2 配置 mlir 环境

```bash
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest

cd /workspace
# 下载编译好的mlir包
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/mlir_club/tpu-mlir_v1.0.0.dev-ed1fc7c49-20260618.tar.gz
tar zxf tpu-mlir_v1.0.0.dev-ed1fc7c49-20260618.tar.gz
cd tpu-mlir_v1.0.0.dev-ed1fc7c49-20260618
source ./envsetup.sh  #激活环境变量
```

### 1.3 编译模型

```bash
# E2B 
llm_convert.py -m gemma-4-E2B-it-AWQ-4bit -s 4096 --max_input_length 2560 -q w4f16 -c bm1684x -o bmodel/ --audio_length 750

# E4B
llm_convert.py -m gemma-4-E4B-it-AWQ-4bit -s 2048 --max_input_length 1024 -q w4f16 -c bm1684x -o bmodel/ --audio_length 750
```

参数说明：
- `-m`: 源模型路径
- `-s`: 序列长度（seq_length）
- `--max_input_length`: 最大输入长度（prefill 阶段有效 token 数）
- `-q`: 量化类型（bf16 / w4f16 等）
- `-c`: 目标芯片（bm1684x / bm1688 等）
- `-o`: 输出目录
- `--audio_length`: 音频最大 token 数，默认 750

## 2. 配置环境

```bash
sudo apt-get update
sudo apt-get install pybind11-dev

pip3 install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 transformers==5.5.4 Jinja2==3.1.2 librosa ffmpeg-python av
```

## 3. 编译 demo

```bash
cd python_demo
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

## 4. 运行 demo

### CLI demo（交互式）

```bash
python3 pipeline.py -m your_bmodel_path --embed_path per_layer_token_embd_e2b.bin -c ../config -d 0
```

运行后交互式输入问题，可选附带图片/视频/音频路径。

### 支持的输入类型

| 输入类型 | 支持的文件格式 |
|---------|-------------|
| 文本 | 直接输入问题 |
| 图片 | jpg, jpeg, png, gif, bmp, webp |
| 视频 | mp4, avi, mov, mkv, flv, wmv |
| 音频 | wav, mp3, flac, ogg, m4a, aac, wma |

### 注意

各模态 token 占用：
- **图片**：token 数 = `(resize后H // 16) × (resize后W // 16) // 9`。任意分辨率的图片会保持宽高比 resize 到最大且不超过 645120 pixels， 即 280 tokens。例如正方形图片 resize 后分辨率为 768×768，对应 256 tokens
- **视频**：固定 32 帧，每帧最多 70 tokens，即161280 pixels，共 2240 tokens。超过 32 帧会均匀采样到 32 帧，不足 32 帧会重复最后一帧补齐到 32 帧。`max_input_length` 低于 2240 的模型不支持视频输入
- **音频**：每秒 25 tokens，最多支持 30 秒音频（750 tokens），编译时 `--audio_length` 参数控制上限

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model_path` | bmodel 文件路径 | 必填 |
| `-c, --config_path` | tokenizer/config 目录路径 | `../config` |
| `-d, --devid` | 使用的设备 ID | `0` |

### 示例

**文本问答：**
```
Question: Write a short joke about saving RAM.
Image, Video, or Audio Path: (直接回车跳过)
```

**图片理解：**
```
Question: What is shown in this image?
Image, Video, or Audio Path: test.jpg
```

**视频理解：**
```
Question: Describe this video.
Image, Video, or Audio Path: test.mp4
```

**音频理解：**
```
Question: Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.
Image, Video, or Audio Path: asr_zh.wav
```