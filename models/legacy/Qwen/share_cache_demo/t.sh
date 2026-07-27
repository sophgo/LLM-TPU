#!/bin/bash
set -ex
max_pos_len=8192 # length of the rotary position embedding; must be set to the same value so that block_cache and block weights can be merged
generation_mode=default # decoding mode
embedding_mode=binary # when set to binary, the bmodel does not contain the embedding; it is stored on disk instead
dynamic=1 # enable dynamic shapes in the prefill stage

seq_length_list=8192,7168,6144,5120,4096,3072,2048,1024 # input length + output length must not exceed seq_length
share_length_list=8192,7168,6144,5120,4096,3072,2048,1024 # input length share_length
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
