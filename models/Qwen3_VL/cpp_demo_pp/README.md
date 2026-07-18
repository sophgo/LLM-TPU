# Qwen3-VL C++ Multi-Device Inference Demo

This project is a C++ multi-device inference demo for the Qwen3-VL multimodal large model. It supports distributing the model across multiple TPU devices for inference, and supports image, video, and pure-text interaction.

## Models

``` shell
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3vl_multi/qwen3-vl-4b-instruct-awq-4bit_w4bf16_seq2048_bm1684x_6dev_dynamic_a_block.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3vl_multi/qwen3-vl-4b-instruct-awq-4bit_w4bf16_seq2048_bm1684x_6dev_dynamic_b_block.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3vl_multi/qwen3-vl-4b-instruct-awq-4bit_w4bf16_seq2048_bm1684x_6dev_dynamic_c_block.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3vl_multi/qwen3-vl-4b-instruct-awq-4bit_w4bf16_seq2048_bm1684x_6dev_dynamic_d_block.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3vl_multi/qwen3-vl-4b-instruct-awq-4bit_w4bf16_seq2048_bm1684x_6dev_embed_vit.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3vl_multi/qwen3-vl-4b-instruct-awq-4bit_w4bf16_seq2048_bm1684x_6dev_dynamic_lmhead.bmodel
```

## Build

### Option 1: Use System OpenCV

``` shell
sudo apt update
sudo apt install libopencv-dev

mkdir build && cd build
cmake .. && make
```

### Option 2: Use Sophon OpenCV (/opt/sophon/sophon-opencv-latest)

Edit `CMakeLists.txt` and uncomment the following line:
```cmake
set(SOPHON_OPENCV TRUE)
```

Then build:
``` shell
mkdir build && cd build
cmake .. && make
```

## Run

```shell
./pipeline -m <model_path> -c <config_path> [options]
```

### Parameter Description

| Parameter | Description | Default |
|------|------|--------|
| `-m`, `--model` | Directory containing the bmodel files (required) | — |
| `-c`, `--config` | Configuration directory; must contain `tokenizer.json` (required) | — |
| `-d`, `--devid` | TPU device IDs; separate multiple devices with commas | `0` |
| `-r`, `--video_ratio` | Video sampling ratio | `0.25` |
| `-f`, `--video_fps` | Video sampling frame rate | `1.0` |
| `-s`, `--do_sample` | Enable sampling generation (greedy decoding by default) | `false` |

### Examples

```shell
# Run on a single device
./pipeline -m /path/to/bmodel -c /path/to/config

# Run on multiple devices (using device 0 and device 1)
./pipeline -m /path/to/bmodel -c /path/to/config -d 0,1

# Enable sampling
./pipeline -m /path/to/bmodel -c /path/to/config -s
```

### Interactive Commands

After running, an interactive conversation starts. Enter your question and media file path in turn (leave empty for pure-text conversation):
- Enter `q` / `quit` / `exit`: exit the program
- Enter `clear` / `new` / `c`: clear history and start a new conversation
