#!/bin/bash
set -ex
max_length=8192
share_length=6144
unshare_length=1536
seq_length=8192
python export_onnx.py --model_path /workspace/models/Qwen2-7B-Instruct/ --device cpu --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread 16 --max_pos_len $max_length
./compile.sh --mode int4 --name qwen2-7b --share_length $share_length --addr_mode io_alone --unshare_length $unshare_length

share_length=6144
unshare_length=1024
seq_length=7680
python export_onnx.py --model_path /workspace/models/Qwen2-7B-Instruct/ --device cpu --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread 16 --max_pos_len $max_length
./compile.sh --mode int4 --name qwen2-7b --share_length $share_length --addr_mode io_alone --unshare_length $unshare_length

share_length=1248
unshare_length=0
seq_length=1248
python export_onnx.py --model_path /workspace/models/Qwen2-7B-Instruct/ --device cpu --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread 16 --max_pos_len $max_length
./compile.sh --mode int4 --name qwen2-7b --share_length $share_length --addr_mode io_alone --unshare_length $unshare_length

#share_length=6144
#unshare_length=1024
#seq_length=8192
#python export_onnx.py --model_path /workspace/models/Qwen2-7B-Instruct/ --device cpu --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread 16 --max_pos_len $seq_length
#./compile.sh --mode int4 --name qwen2-7b --share_length $share_length --addr_mode io_alone --unshare_length $unshare_length
#
#share_length=6144
#unshare_length=2048
#seq_length=8192
#python export_onnx.py --model_path /workspace/models/Qwen2-7B-Instruct/ --device cpu --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread 16 --max_pos_len $seq_length
#./compile.sh --mode int4 --name qwen2-7b --share_length $share_length --addr_mode io_alone --unshare_length $unshare_length
