#!/bin/bash
set -ex
max_pos_len=6912
generation_mode=default

seq_length=3712
share_length=3200
unshare_length=0
python export_onnx.py --model_path /workspace/models/Qwen-7B-Chat/ --device cpu --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread 16 --max_pos_len $max_pos_len --generation_mode $generation_mode
./compile.sh --mode int4 --name qwen-7b --share_length $share_length --addr_mode io_alone --unshare_length $unshare_length --seq_length $seq_length --generation_mode $generation_mode

seq_length=6912
share_length=6400
unshare_length=0
python export_onnx.py --model_path /workspace/models/Qwen-7B-Chat/ --device cpu --share_length $share_length --unshare_length $unshare_length --seq_length $seq_length --num_thread 16 --max_pos_len $max_pos_len --generation_mode $generation_mode
./compile.sh --mode int4 --name qwen-7b --share_length $share_length --addr_mode io_alone --unshare_length $unshare_length --seq_length $seq_length --generation_mode $generation_mode
