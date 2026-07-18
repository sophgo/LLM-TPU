# MiniCPM-V-4.6

This project deploys the multimodal large model [MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) on BM1684X/BM1688. The model is converted into a bmodel through the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler and deployed to the PCIE environment or SoC environment.

The model supports image and video recognition, and a Python version demo is provided.

This document covers how to compile the bmodel and how to run the bmodel in the BM1684X/BM1688 environment. The bmodel compilation step can be skipped; just download it directly from the following links:

``` shell
# =============== 1684x =====================
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm-v-4.6_bf16_seq2048_bm1684x_1dev_dynamic_20260630_105643.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm-v-4.6_bf16_seq2048_bm1688_2core_dynamic_20260630_155028.bmodel
```

## Model Architecture

MiniCPM-V-4.6 has 1.3B parameters and consists of two parts: a vision encoder and a text model:

- **Text model**: Same architecture as Qwen3.5
- **Vision encoder**: SigLIP ViT + Merger downsampling, supporting two modes:
  - `16x` (default): two stages of 2×2 merging, 16x downsampling in total, fewer tokens and faster inference
  - `4x`: one stage of 2×2 merging, 4x downsampling, retaining more visual details

## Compile the bmodel

This section describes how to compile the model into a bmodel.

#### 1. Download the Model

``` shell
git clone https://huggingface.co/openbmb/MiniCPM-V-4.6
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
# max_input_length specifies the maximum input length; if not specified, the length specified by -s is used
llm_convert.py -m /workspace/MiniCPM-V-4.6 -s 2048 --max_input_length 1024 -q bf16 -c bm1684x -o minicpm_v4_6 --max_pixels 448,448
```
After compilation, `minicpm-v-4.6-xxx.bmodel` and `config` are generated in the specified directory.

## Compile and Run the Program (python)

### 1. Environment Preparation

A python3.10 environment is required. If it is not available, refer to [this document](https://github.com/sophgo/sophon-demo/blob/release/docs/FAQ.md#13-se7%E5%AE%89%E8%A3%85python310) to install it.

``` shell
sudo apt-get update
sudo apt-get install pybind11-dev

pip3 install torch==2.6.0 torchvision==0.21.0 transformers==5.7.0
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
python3 pipeline.py -m minicpm-v-4.6.bmodel -c ../config

# Single inference mode (image)
python3 pipeline.py -m minicpm-v-4.6.bmodel -c ../config \
    --prompt "Describe this image" --media_path test.jpg

# Single inference mode (video)
python3 pipeline.py -m minicpm-v-4.6.bmodel -c ../config \
    --prompt "Describe what happened in the video" --media_path test.mp4
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
| `-p, --prompt` | None | Enter single inference mode when specified |
| `-t, --prompt_file` | None | Load the prompt from a file |
| `--media_path` | Empty | Image or video path, used together with `--prompt` |

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