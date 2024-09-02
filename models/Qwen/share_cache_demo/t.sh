#!/bin/bash
set -ex
max_pos_len=8192
generation_mode=default
embedding_mode=binary
dynamic=1

seq_length=8192
share_length=8192
unshare_length=0
python export_onnx.py --model_path /workspace/models/Qwen-7B-Chat/ --device cpu --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread 16 --max_pos_len $max_pos_len --generation_mode $generation_mode --embedding_mode $embedding_mode

seq_length=4352
share_length=4352
unshare_length=0
python export_onnx.py --model_path /workspace/models/Qwen-7B-Chat/ --device cpu --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread 16 --max_pos_len $max_pos_len --generation_mode $generation_mode --embedding_mode $embedding_mode


seq_length_list=8192,4352
share_length_list=8192,4352
unshare_length_list=0,0
./compile_multi.sh --mode int4 --name qwen-7b --share_length_list $share_length_list --addr_mode io_alone --unshare_length_list $unshare_length_list --seq_length_list $seq_length_list --generation_mode $generation_mode --dynamic $dynamic --embedding_mode $embedding_mode

model_tool --encrypt -model qwen-7b.bmodel -net block_0 -lib ../../Qwen2/share_cache_demo/build/libcipher.so -o encrypted.bmodel
