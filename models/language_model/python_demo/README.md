# language_model

## 目录
- [Language Model](#language-model)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 目录结构](#3-目录结构)
  - [4. 运行](#4-运行)
    - [4.1 环境准备](#41-环境准备)
    - [4.2 模型下载与运行](#42-模型下载与运行)
  - [5. 程序性能测试](#5-程序性能测试)

## 1. 简介
本仓库提供了一个通用的大语言模型（Large Language Model）例程，支持在SOPHON BM1684X上进行推理测试。该例程适用于多种语言模型，能够处理文本生成、文本理解等任务。

## 2. 特性
* 支持BM1684X(x86/arm PCIe、SE7)
* 支持INT8、INT4模型编译和推理
* 支持Python例程

## 3. 目录结构
```bash
├── CMakeLists.txt                  # CMakeLists编译文件
├── chat.cpp                        # python依赖的后端cpp文件
├── pipeline.py                     # python推理脚本
├── README.md                       # 例程指南
```

## 4. 运行

### 4.1 环境准备
```bash
git clone https://github.com/sophgo/LLM-TPU.git
pip3 install dfss transformers==4.45.1 pybind11[global] Jinja2
sudo apt install zip

cd LLM-TPU/models/language_model/python_demo
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

### 4.2 模型下载与运行

#### DeepSeek-R1-Distill-Qwen系列
下载`deepseek-r1-distill-qwen-1.5b`模型，并运行：
```bash
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-distill-qwen-1-5b.zip
unzip deepseek-r1-distill-qwen-1-5b.zip
python3 pipeline.py --devid 0 --dir_path ./deepseek-r1-distill-qwen-1-5b/
```

下载`deepseek-r1-distill-qwen-7b`模型，并运行：
```bash
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-distill-qwen-7b.zip
unzip deepseek-r1-distill-qwen-7b.zip
python3 pipeline.py --devid 0 --dir_path ./deepseek-r1-distill-qwen-7b/
```

下载`deepseek-r1-distill-qwen-14b`模型，并运行：
```bash
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-distill-qwen-14b-seq512.zip
unzip deepseek-r1-distill-qwen-14b-seq512.zip
python3 pipeline.py --devid 0 --dir_path ./deepseek-r1-distill-qwen-14b/
```

## 5. 程序性能测试
LLM性能

|   测试平台   |           测试模型              | 量化方式 | 模型长度 | first token latency(s) | token per second(tokens/s) |
| ----------- | ------------------------------ | -------- | -------- | --------------------- | -------------------------- |
| SE7-32      | deepseek-r1-distill-qwen-1.5b  | INT4     | 8192     | 5.431                 | 28.878                     |
| SE7-32      | deepseek-r1-distill-qwen-7b    | INT4     | 2048     | 2.939                 | 10.600                     |
| SE7-32      | deepseek-r1-distill-qwen-14b   | INT4     | 512      | 1.400                 | 5.564                      |

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. SE7-32的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 3. 这里使用的SDK版本是BM1684X V24.04.01；

## 6. 常见问题及解答

请参考[LLM-TPU常见问题及解答](../../../docs/FAQ.md)