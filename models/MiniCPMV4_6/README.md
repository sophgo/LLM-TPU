# MiniCPM-V-4.6 / MiniCPM-V-4.7

This project deploys the multimodal large models [MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) and [MiniCPM-V-4.7](https://huggingface.co/openbmb/MiniCPM-V-4.7) on BM1684X/BM1688. The model is converted into a bmodel through the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler and deployed to the PCIE environment or SoC environment.

Both models support image and video recognition, and a Python version demo is provided.

> ⚠️ **The `config/` directory shipped in this repository is for MiniCPM-V-4.6 only.**
> To run MiniCPM-V-4.7 you must obtain its own `config/` — either by compiling it
> yourself (see below) or by downloading the matching bmodel + config bundle from
> dfss. Do NOT reuse the 4.6 config with a 4.7 bmodel (the 4.7 pipeline needs
> `image_start_id` / `mm_token_type_ids` / Canvas M-RoPE, none of which exist in
> the 4.6 config). The 4.6 config works for both 4.6 and 4.7 pipelines when paired
> with a 4.6 bmodel.

This document covers how to compile the bmodel and how to run the bmodel in the BM1684X/BM1688 environment. The bmodel compilation step can be skipped; just download it directly from the following links:

``` shell
# =============== MiniCPM-V-4.6 (bf16) =====================
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm-v-4.6_bf16_seq2048_bm1684x_1dev_dynamic_20260630_105643.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm-v-4.6_bf16_seq2048_bm1688_2core_dynamic_20260630_155028.bmodel

# =============== MiniCPM-V-4.7 (GPTQ W4A16) =====================
# NOTE: download the 4.7 config/ from the same model release page or compile it yourself;
# the in-repo config/ is 4.6-only (see warning above).
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm-v-4.7-1b-gptq_w4bf16_seq2048_bm1684x_1dev_dynamic_20261029_181158.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm-v-4.7-1b-gptq_w4bf16_seq2048_bm1688_2core_dynamic_20261030_113652.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm-v-4.7-1b-gptq_w4bf16_seq2048_cv84x6_4core_dynamic_20261030_113824.bmodel
```

## Model Architecture

MiniCPM-V-4.6 has 1.3B parameters and MiniCPM-V-4.7 has 1B parameters. Both consist of two parts: a vision encoder and a text model:

- **Text model**: Same architecture as Qwen3.5
- **Vision encoder**: SigLIP ViT + Merger downsampling, supporting two modes:
  - `16x` (default): two stages of 2×2 merging, 16x downsampling in total, fewer tokens and faster inference
  - `4x`: one stage of 2×2 merging, 4x downsampling, retaining more visual details

### MiniCPM-V-4.7 vs 4.6

| Feature | 4.6 | 4.7 |
|---------|-----|-----|
| Parameters | 1.3B | 1B (GPTQ W4A16) |
| lm_head | Tied to embedding | Untied (separate weight) |
| M-RoPE | Standard 3D | Canvas M-RoPE (host-side) |
| ViT merger residual | Raster-order | Window-order |

The TPU-MLIR converter handles both versions. The single `pipeline.py` works with either — it auto-detects the version from the config's `mrope_mode` field and uses Canvas M-RoPE for 4.7 or plain 3-D M-RoPE for 4.6. Always pair a bmodel with the config it was compiled with (both models share the tokenizer, but `config.json` / `preprocessor_config.json` differ).

## Compile the bmodel

This section describes how to compile the model into a bmodel.

#### 1. Download the Model

``` shell
# MiniCPM-V-4.6
git clone https://huggingface.co/openbmb/MiniCPM-V-4.6

# MiniCPM-V-4.7 (GPTQ version recommended)
git clone https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-GPTQ
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
source ./envsetup.sh  #activate environment variables
./build.sh #compile mlir
```

#### 4. Compile the Model to Generate the bmodel

``` shell
# MiniCPM-V-4.6 (bf16)
llm_convert.py -m /workspace/MiniCPM-V-4.6 -s 2048 --max_input_length 1024 -q bf16 -c bm1684x -o minicpm_v4_6 --max_pixels 448,448

# MiniCPM-V-4.7 (GPTQ W4A16, follows source quantization)
llm_convert.py -m /workspace/MiniCPM-V-4.7-1B-GPTQ -s 2048 --max_input_length 1024 -q w4bf16 -c bm1684x -o minicpm_v4_7 --max_pixels 448,448
```
After compilation, `minicpm-v-4.x-xxx.bmodel` and `config` are generated in the specified directory.

## Compile and Run the Program (python)

### 1. Environment Preparation

A python3.10 environment is required. If it is not available, refer to [this document](https://github.com/sophgo/sophon-demo/blob/release/docs/FAQ.md#13-se7%E5%AE%89%E8%A3%85python310) to install it.

``` shell
sudo apt-get update
sudo apt-get install pybind11-dev

# For video support, torchvision 0.21+ is required (do NOT install torchcodec)
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

`pipeline.py` auto-detects MiniCPM-V-4.6 vs 4.7 from the config (`mrope_mode == "canvas"` → 4.7). Just point `-m` and `-c` at the matching bmodel + config.

``` shell
# ========== MiniCPM-V-4.6 (uses the in-repo config/) ==========
# Interactive mode
python3 pipeline.py -m minicpm-v-4.6.bmodel -c ../config

# Single inference mode (image)
python3 pipeline.py -m minicpm-v-4.6.bmodel -c ../config \
    --prompt "Describe this image @test.jpg"

# Single inference mode (video)
python3 pipeline.py -m minicpm-v-4.6.bmodel -c ../config \
    --prompt "Describe what happened in the video @test.mp4"

# ========== MiniCPM-V-4.7 (needs 4.7 config, NOT the in-repo 4.6 config) ==========
# Interactive mode
python3 pipeline.py -m minicpm-v-4.7.bmodel -c ../config

# Single inference mode (image)
python3 pipeline.py -m minicpm-v-4.7.bmodel -c ../config \
    --prompt "Describe this image @test.jpg"

# Single inference mode (video)
python3 pipeline.py -m minicpm-v-4.7.bmodel -c ../config \
    --prompt "Describe what happened in the video @test.mp4"
```

### CLI Parameters

| Parameter | Default | Description |
|------|--------|------|
| `-m, --model_path` | Required | Path to the bmodel file |
| `-c, --config_path` | `../config` | Directory of the processor config files |
| `-d, --devid` | `0` | Device ID |
| `--downsample_mode` | `16x` | ViT downsampling mode, either `4x` or `16x` |
| `--max_slice_nums` | None | Maximum number of image slices. If not specified, the default is 36 for images and 1 for videos; if specified, it applies uniformly |
| `--max_num_frames` | `16` | Maximum number of sampled video frames |
| `-p, --prompt` | None | Enter single inference mode when specified; include `@<path>` to attach an image or video, or to read prompt text from a `.txt`/`.md` file |

## Performance

MiniCPM-V-4.7 (GPTQ W4A16, seq 2048, max_pixels 448×448) measured on the `test.jpg` sample (448×448, 1024 patches).

| Platform | Cores | Text TPS (tokens/s) | Image 16x FTL (s) | Image 16x TPS | Image 4x FTL (s) | Image 4x TPS |
|----------|-------|--------------------:|------------------:|--------------:|-----------------:|-------------:|
| BM1684X (PCIE/SoC) | 1 | 54.9 | 0.54 | 54.9 | 1.02 | 54.9 |
| BM1688 (SoC) | 2 | 29.0 | 1.83 | 29.0 | 3.22 | 29.0 |
| CV84X6 (SoC) | 4 | 80.4 | 1.23 | 80.3 | 2.05 | 80.4 |

Notes:
- **FTL** = first-token latency (prompt processing + first decode); **TPS** = steady-state decode tokens/sec.
- Decode TPS is constant per platform regardless of input mode (text/16x/4x) — only prefill (FTL) grows with more visual tokens.
- CV84X6 (4-core) has the highest throughput; BM1684X is fastest per-core; BM1688 is a dual-core cost-optimized part.
- Image FTL differences reflect prefill token count: 16x → 64 visual tokens, 4x → 256 visual tokens (see FAQ below).

## Advanced Applications

Refer to [Qwen3.5 README](../Qwen3_5/README.md).

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
