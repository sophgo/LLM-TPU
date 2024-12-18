#!/bin/bash
set -ex

# compile
# 配置环境
pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
pip3 install transformers_stream_generator einops tiktoken accelerate transformers==4.41.2 peft

max_rank_num=64 # 开启lora后，外挂的lora分支的秩
max_embedding_rank_num=64 # 开启lora embedding后，外挂的lora embedding分支的秩

model_path="/workspace/models/Qwen2-7B-Instruct/" # 训练的pytorch基座模型的路径
lib_path="../share_cache_demo/build/libcipher.so" # 加解密so的路径
lora_path="saves_lora/lora_sft_qwen2_unpretrained_init/" # 微调的lora模型的路径
lora_embedding_path="saves_lora/lora_sft_qwen2_unpretrained_init_embedding/" # 微调的lora模型的路径
device="cpu"
num_thread=16

python export_abnormal.py \
  --model_path $model_path \
  --device $device \
  --num_thread $num_thread \
  --lib_path $lib_path \
  --lora_path $lora_path \
  --lora_embedding_path $lora_embedding_path \
  --max_rank_num $max_rank_num \
  --max_embedding_rank_num $max_embedding_rank_num

# run