#!/bin/bash
set -ex

# 配置环境
pip3 install torch==2.0.1 transformers_stream_generator einops tiktoken accelerate transformers==4.41.2
cp files/Qwen2-7B-Instruct/* /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/

max_pos_len=10240 # 旋转位置编码的长度，设置为同一个值，才能将block_cache和block权重合并
generation_mode=default # 解码模式
embedding_mode=binary # 设置为binary时，bmodel中不包含embedding，而是放到硬盘
dynamic=1 # prefill阶段开启动态
max_rank_num=64 # 开启lora后，外挂的lora分支的秩

seq_length_list=10240,8192,7168,6144,5120,4096,3072,2048,1024 # 输入长度 + 输出长度不能超过seq_length
share_length_list=8320,8192,7168,6144,5120,4096,3072,2048,1024 # 输入长度share_length
unshare_length_list=0,0,0,0,0,0,0,0,0
model_path="/workspace/models/Qwen2-7B-Instruct/"
lib_path="../share_cache_demo/build/libcipher.so"
lora_path="saves_lora/lora_sft_qwen2_unpretrained_init/"
device="cpu"
num_thread=16

# Convert comma-separated lists to arrays
IFS=',' read -r -a seq_lengths <<< "$seq_length_list"
IFS=',' read -r -a share_lengths <<< "$share_length_list"
IFS=',' read -r -a unshare_lengths <<< "$unshare_length_list"

for i in "${!seq_lengths[@]}"; do
  seq_length=${seq_lengths[$i]}
  share_length=${share_lengths[$i]}
  unshare_length=${unshare_lengths[$i]}
  python export_onnx.py --model_path $model_path --device $device --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread $num_thread --max_pos_len $max_pos_len --generation_mode $generation_mode --embedding_mode $embedding_mode --lora_path $lora_path --max_rank_num $max_rank_num
done

./compile_multi.sh --mode int4 --name qwen2-7b --share_length_list $share_length_list --addr_mode io_alone --unshare_length_list $unshare_length_list --seq_length_list $seq_length_list --generation_mode $generation_mode --dynamic $dynamic --embedding_mode $embedding_mode

model_tool --encrypt -model qwen2-7b.bmodel -net block_0 -lib $lib_path -o encrypted.bmodel