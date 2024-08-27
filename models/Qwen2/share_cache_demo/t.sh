#!/bin/bash
set -ex
max_pos_len=10240
generation_mode=default
embedding_mode=default

seq_length=10240
share_length=8192
unshare_length=0
python export_onnx.py --model_path /workspace/models/Qwen2-7B-Instruct/ --device cpu --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread 16 --max_pos_len $max_pos_len --generation_mode $generation_mode --embedding_mode $embedding_mode
./compile.sh --mode int4 --name qwen2-7b --share_length $share_length --addr_mode io_alone --unshare_length $unshare_length --seq_length $seq_length --generation_mode $generation_mode --dynamic 1 --embedding_mode $embedding_mode

weight_1=$(model_tool --info qwen2-7b_int4_share${share_length}_unshare${unshare_length}_seq${seq_length}_1dev_dyn.bmodel | grep -oP 'weight: \K\d+' | head -n 1)

model_tool --encrypt -model qwen2-7b_int4_share${share_length}_unshare${unshare_length}_seq${seq_length}_1dev_dyn.bmodel -net block_0 -lib ../../Qwen2/share_cache_demo/build/libcipher.so -o qwen2-7b_int4_share${share_length}_unshare${unshare_length}_seq${seq_length}_1dev_dyn_encrypted.bmodel

seq_length=5120
share_length=4096
unshare_length=0
python export_onnx.py --model_path /workspace/models/Qwen2-7B-Instruct/ --device cpu --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread 16 --max_pos_len $max_pos_len --generation_mode $generation_mode --embedding_mode $embedding_mode
./compile.sh --mode int4 --name qwen2-7b --share_length $share_length --addr_mode io_alone --unshare_length $unshare_length --seq_length $seq_length --generation_mode $generation_mode --dynamic 1 --embedding_mode $embedding_mode

weight_0=$(model_tool --info qwen2-7b_int4_share${share_length}_unshare${unshare_length}_seq${seq_length}_1dev_dyn.bmodel | grep -oP 'weight: \K\d+' | head -n 1)

model_tool --encrypt -model qwen2-7b_int4_share${share_length}_unshare${unshare_length}_seq${seq_length}_1dev_dyn.bmodel -net block_0 -lib ../../Qwen2/share_cache_demo/build/libcipher.so -o qwen2-7b_int4_share${share_length}_unshare${unshare_length}_seq${seq_length}_1dev_dyn_encrypted.bmodel



if [ "$weight_0" -ne "$weight_1" ]; then
    echo "Error: weight_0 ($weight_0) is not equal to weight_1 ($weight_1)"
    exit 1
else
    echo "weight_0 ($weight_0) is equal to weight_1 ($weight_1)"
fi
