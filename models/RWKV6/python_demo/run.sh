#!/bin/bash

# 创建并进入 build 目录
mkdir -p build
cd build

# 运行 cmake 和 make 命令
cmake ..
make

# 将生成的 cpython 文件复制到上一级目录
cp *cpython* ..

# 返回上一级目录
cd ..

# 进入 RWKV6 的 python_demo 目录
cd models/RWKV6/python_demo/

# 运行 Python 脚本
python pipeline.py --model_path /data/work/LLM-TPU-RWKV-dev/bmodels/rwkv6-1b5_bf16_1dev.bmodel --tokenizer_path ./rwkv_vocab_v20230424.txt --devid 0 --generation_mode greedy

