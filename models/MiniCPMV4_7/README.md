# MiniCPM-V-4.7

This project deploys the multimodal models [MiniCPM-V-4.7-1B-AWQ](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-AWQ) and [MiniCPM-V-4.7-35B-A3B-AWQ](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-AWQ) on BM1684X/BM1688/CV84X6. Each model is converted into a bmodel through the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler and deployed to the PCIE environment or SoC environment.

Both models support image and video recognition. A Python demo is provided and is shared by the variants; use the bmodel and the `config` directory produced by the same compile.

This document covers how to compile the bmodel and how to run the bmodel in the BM1684X/BM1688/CV84X6 environment. The bmodel compilation step can be skipped; just download it directly from the following links:

``` shell
# =============== MiniCPM-V-4.7-1B-GPTQ (w4bf16) =====================
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm-v-4.7-1b-gptq_w4bf16_seq2048_bm1684x_1dev_dynamic_20261029_181158.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm-v-4.7-1b-gptq_w4bf16_seq2048_bm1688_2core_dynamic_20261030_113652.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm-v-4.7-1b-gptq_w4bf16_seq2048_cv84x6_4core_dynamic_20261030_113824.bmodel

# =============== MiniCPM-V-4.7-35B-A3B-GPTQ (w4bf16) ================
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm-v-4.7-35b-a3b-gptq_w4bf16_seq2048_cv84x6_4core_dynamic_20261103_112343.bmodel
```

## Compile the LLM Model

This section describes how to compile the model into a bmodel.

#### 1. Download the Model

``` shell
git lfs install
# 1B
git clone https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-AWQ
# 35B-A3B
git clone https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-AWQ
# make sure the safetensors are fully downloaded (git lfs pull), not LFS pointer stubs
```

#### 2. Download docker and Start the Container

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
The following assumes that the environment is in the `/workspace` directory of docker.

#### 3. Download the `TPU-MLIR` Code and Compile It

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  # activate the environment variables
./build.sh # compile mlir
```

#### 4. Compile the Model to Generate the bmodel

``` shell
# max_input_length specifies the maximum input length; if not specified, the length specified by -s is used
llm_convert.py -m /workspace/MiniCPM-V-4.7-1B-AWQ -s 2048 --max_input_length 1024 -q auto -c bm1684x -o minicpm_v4_7_1b
```
After compilation, `minicpm-v-4.7-xxx.bmodel` and `config` are generated in the specified directory.

> **Note**: The bmodel is always paired with the `config/` directory produced by the same compile. Do not mix a bmodel with a config from a different model or revision.

## Compile and Run the Program (python)

### 1. Environment Preparation

A python3.10 environment is required. If it is not available, refer to [this document](https://github.com/sophgo/sophon-demo/blob/release/docs/FAQ.md#13-se7%E5%AE%89%E8%A3%85python310) to install it.

``` shell
sudo apt-get update
sudo apt-get install pybind11-dev

pip3 install torch==2.6.0 torchvision==0.21.0 transformers==5.18.0
```

### 2. Compile the Library Files

Compile the C++ library files to generate `chat.cpython*.so`:

``` shell
cd python_demo
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

### 3. Run

``` shell
# Interactive mode
python3 pipeline.py -m minicpm-v-4.7.bmodel -c ../config

# Single inference mode (image)
python3 pipeline.py -m minicpm-v-4.7.bmodel -c ../config \
    --prompt "Describe this image @test.jpg"

# Single inference mode (video)
python3 pipeline.py -m minicpm-v-4.7.bmodel -c ../config \
    --prompt "Describe what happened in the video @test.mp4"
```

## Performance

MiniCPM-V-4.7-1B-GPTQ (w4bf16, seq 2048, max_pixels 448×448) measured on the `test.jpg` sample (448×448, 1024 patches).

| Platform | Cores | Text TPS (tokens/s) | Image 16x FTL (s) | Image 16x TPS | Image 4x FTL (s) | Image 4x TPS |
|----------|-------|--------------------:|------------------:|--------------:|-----------------:|-------------:|
| BM1684X (PCIE/SoC) | 1 | 54.9 | 0.54 | 54.9 | 1.02 | 54.9 |
| BM1688 (SoC) | 2 | 29.0 | 1.83 | 29.0 | 3.22 | 29.0 |
| CV84X6 (SoC) | 4 | 80.4 | 1.23 | 80.3 | 2.05 | 80.4 |

MiniCPM-V-4.7-35B-A3B-GPTQ (w4bf16, seq 2048, max_pixels 448×448) measured on the `test.jpg` sample.

| Platform | Cores | Text TPS (tokens/s) | Image 16x FTL (s) | Image 16x TPS | Image 4x FTL (s) | Image 4x TPS |
|----------|-------|--------------------:|------------------:|--------------:|-----------------:|-------------:|
| CV84X6 (SoC) | 4 | 12.6 | 5.04 | 12.6 | 11.22 | 12.5 |

Notes:
- **FTL** = first-token latency (prompt processing + first decode); **TPS** = steady-state decode tokens/sec.
- Decode TPS is constant per platform regardless of input mode (text/16x/4x) — only prefill (FTL) grows with more visual tokens.
- CV84X6 (4-core) has the highest throughput; BM1684X is fastest per-core; BM1688 is a dual-core cost-optimized part.
- Image FTL differences reflect prefill token count: 16x → 64 visual tokens, 4x → 256 visual tokens.

## FAQ

#### How many tokens does one image occupy?

Formula: $tokens = \frac{h_{patches} \times w_{patches}}{merge\_size^2}$

Where $h_{patches} = \frac{height}{14}$, $w_{patches} = \frac{width}{14}$.

Taking a 448×448 image as an example:
- patches = 32 × 32 = 1024
- 16x mode: 1024 / 16 = **64 tokens**
- 4x mode: 1024 / 4 = **256 tokens**

High-resolution images are processed by slicing, and `max_slice_nums` controls the maximum number of slices.

#### How many tokens does a video occupy?

Each video frame is processed independently, with `max_slice_nums=1` (no slicing).

Taking the 16x mode with 448×448 per frame as an example, each frame occupies 64 tokens.

A 16-frame video: $64 × 16 = 1024$ tokens

#### What is the maximum number of frames supported for a video?

| Downsampling mode | Tokens per frame | Maximum frames (SEQLEN=2048) |
|-----------|-----------|----------------------|
| 16x | ~64 | ~25 |
| 4x | ~256 | ~6 |

The actual available number of frames also needs to subtract the tokens occupied by the prompt and the output.

#### Video decoding fails with "torchcodec is not installed"?

Install `torchvision==0.21.0` (which includes `torchvision.io.read_video`). Do NOT install `torchcodec` — it requires FFmpeg libraries not present on typical TPU hosts and may hang during import.

#### "ViT input has N patches, exceeding the compiled MAX_PATCHES"?

Each ViT forward handles at most one 448×448-equivalent slice (1024 patches). If the preprocessor produces a single slice larger than this (e.g. a very wide or tall image that isn't split), the demo aborts with this error instead of corrupting device memory. Use a smaller `--max_pixels` at compile time to raise the limit, or pre-split the image.
