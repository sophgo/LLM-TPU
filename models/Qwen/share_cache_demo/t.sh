#!/bin/bash
set -ex
max_pos_len=8192
generation_mode=default


seq_length=8192
share_length=4096
unshare_length=4096
python export_onnx.py --model_path /workspace/models/Qwen-7B-Chat/ --device cpu --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread 16 --max_pos_len $max_pos_len --generation_mode $generation_mode
./compile.sh --mode int4 --name qwen-7b --share_length $share_length --addr_mode io_alone --unshare_length $unshare_length --seq_length $seq_length --generation_mode $generation_mode --dynamic 1

weight_1=$(model_tool --info qwen-7b_int4_share${share_length}_unshare${unshare_length}_seq${seq_length}_1dev.bmodel | grep -oP 'weight: \K\d+' | head -n 1)

# model_tool --encrypt -model qwen-7b_int4_share${share_length}_unshare${unshare_length}_seq${seq_length}_1dev.bmodel -net block_0 -lib ../../Qwen2/share_cache_demo/build/libcipher.so -o qwen-7b_int4_share${share_length}_unshare${unshare_length}_seq${seq_length}_1dev_encrypted.bmodel

seq_length=4352
share_length=4096
unshare_length=0
python export_onnx.py --model_path /workspace/models/Qwen-7B-Chat/ --device cpu --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread 16 --max_pos_len $max_pos_len --generation_mode $generation_mode
./compile.sh --mode int4 --name qwen-7b --share_length $share_length --addr_mode io_alone --unshare_length $unshare_length --seq_length $seq_length --generation_mode $generation_mode --dynamic 1

weight_0=$(model_tool --info qwen-7b_int4_share${share_length}_unshare${unshare_length}_seq${seq_length}_1dev.bmodel | grep -oP 'weight: \K\d+' | head -n 1)

# model_tool --encrypt -model qwen-7b_int4_share${share_length}_unshare${unshare_length}_seq${seq_length}_1dev.bmodel -net block_0 -lib ../../Qwen2/share_cache_demo/build/libcipher.so -o qwen-7b_int4_share${share_length}_unshare${unshare_length}_seq${seq_length}_1dev_encrypted.bmodel



if [ "$weight_0" -ne "$weight_1" ]; then
    echo "Error: weight_0 ($weight_0) is not equal to weight_1 ($weight_1)"
    exit 1
else
    echo "weight_0 ($weight_0) is equal to weight_1 ($weight_1)"
fi
