#!/bin/bash
set -ex
max_pos_len=8192 # 旋转位置编码的长度，设置为同一个值，才能将block_cache和block权重合并
generation_mode=default # 解码模式
embedding_mode=binary # 设置为binary时，bmodel中不包含embedding，而是放到硬盘
dynamic=1 # prefill阶段开启动态

seq_length_list=8192,7168,6144,5120,4096,3072,2048,1024 # 输入长度 + 输出长度不能超过seq_length
share_length_list=8192,7168,6144,5120,4096,3072,2048,1024 # 输入长度share_length
unshare_length_list=0,0,0,0,0,0,0,0,0
model_path="/workspace/models/Qwen-7B-Chat/"
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
  python export_onnx.py --model_path $model_path --device $device --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread $num_thread --max_pos_len $max_pos_len --generation_mode $generation_mode --embedding_mode $embedding_mode
done

./compile_multi.sh --mode int4 --name qwen-7b --share_length_list $share_length_list --addr_mode io_alone --unshare_length_list $unshare_length_list --seq_length_list $seq_length_list --generation_mode $generation_mode --dynamic $dynamic --embedding_mode $embedding_mode

model_tool --encrypt -model qwen-7b.bmodel -net block_0 -lib ../../Qwen2/share_cache_demo/build/libcipher.so -o encrypted.bmodel
