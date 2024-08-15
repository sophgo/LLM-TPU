#!/bin/bash
set -ex
seq_lengh=8192
share_length=1248
unshare_length=0
python export_onnx.py --model_path /workspace/models/Qwen-7B-Chat/ --device cpu --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread 16 --max_pos_len $seq_length
./compile.sh --mode int4 --name qwen-7b --share_length $share_length --addr_mode io_alone --unshare_length $unshare_length


share_length=6016
unshare_length=1024
python export_onnx.py --model_path /workspace/models/Qwen-7B-Chat/ --device cpu --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread 16 --max_pos_len $seq_length
./compile.sh --mode int4 --name qwen-7b --share_length $share_length --addr_mode io_alone --unshare_length $unshare_length

share_length=6016
unshare_length=1600
python export_onnx.py --model_path /workspace/models/Qwen-7B-Chat/ --device cpu --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread 16 --max_pos_len $seq_length
./compile.sh --mode int4 --name qwen-7b --share_length $share_length --addr_mode io_alone --unshare_length $unshare_length
