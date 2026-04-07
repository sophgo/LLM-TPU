# Qwen3-VL C++ 多设备推理 Demo

本工程为 Qwen3-VL 多模态大模型的 C++ 多设备推理 Demo，支持将模型分布到多个 TPU 设备上进行推理，支持图片、视频和纯文本交互。

## 模型

``` shell
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3vl_multi/qwen3-vl-4b-instruct-awq-4bit_w4bf16_seq2048_bm1684x_6dev_dynamic_a_block.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3vl_multi/qwen3-vl-4b-instruct-awq-4bit_w4bf16_seq2048_bm1684x_6dev_dynamic_b_block.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3vl_multi/qwen3-vl-4b-instruct-awq-4bit_w4bf16_seq2048_bm1684x_6dev_dynamic_c_block.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3vl_multi/qwen3-vl-4b-instruct-awq-4bit_w4bf16_seq2048_bm1684x_6dev_dynamic_d_block.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3vl_multi/qwen3-vl-4b-instruct-awq-4bit_w4bf16_seq2048_bm1684x_6dev_embed_vit.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3vl_multi/qwen3-vl-4b-instruct-awq-4bit_w4bf16_seq2048_bm1684x_6dev_dynamic_lmhead.bmodel
```

## 编译

### 方式一：使用系统 OpenCV

``` shell
sudo apt update
sudo apt install libopencv-dev

mkdir build && cd build
cmake .. && make
```

### 方式二：使用 Sophon OpenCV（/opt/sophon/sophon-opencv-latest）

修改 `CMakeLists.txt`，取消以下行的注释：
```cmake
set(SOPHON_OPENCV TRUE)
```

然后编译：
``` shell
mkdir build && cd build
cmake .. && make
```

## 运行

```shell
./pipeline -m <model_path> -c <config_path> [options]
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m`, `--model` | bmodel 文件所在目录（必填） | — |
| `-c`, `--config` | 配置文件目录，需包含 `tokenizer.json`（必填） | — |
| `-d`, `--devid` | TPU 设备 ID，多设备用逗号分隔 | `0` |
| `-r`, `--video_ratio` | 视频采样比例 | `0.25` |
| `-f`, `--video_fps` | 视频采样帧率 | `1.0` |
| `-s`, `--do_sample` | 启用采样生成（默认贪心解码） | `false` |

### 示例

```shell
# 单设备运行
./pipeline -m /path/to/bmodel -c /path/to/config

# 多设备运行（使用设备0和设备1）
./pipeline -m /path/to/bmodel -c /path/to/config -d 0,1

# 启用采样
./pipeline -m /path/to/bmodel -c /path/to/config -s
```

### 交互命令

运行后进入交互式对话，依次输入问题和媒体文件路径（留空则为纯文本对话）：
- 输入 `q` / `quit` / `exit`：退出程序
- 输入 `clear` / `new` / `c`：清除历史，开始新对话

