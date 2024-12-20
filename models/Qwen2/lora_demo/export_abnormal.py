#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import os
import argparse

from export_onnx import load_model, convert_total_lora_to_bit, convert_embedding_to_bit

def convert_abnormal():
    for rank in [32, 64, 96]:
        args.max_embedding_rank_num = rank
        args.max_rank_num = rank
        convert_total_lora_to_bit(f"{dir_path}/encrypted_lora_weights_r{rank}.bin", origin_model, args)

    convert_embedding_to_bit(f"{dir_path}/embedding.bin", transformer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='export onnx')
    parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
    parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument('-n', '--num_threads', type=int, default=1, help='The number of threads used for torch if device is cpu')
    parser.add_argument('--lib_path', type=str, default='', help='lib path by user')
    parser.add_argument('--lora_path', type=str, default="", help="path to the lora model")
    parser.add_argument('--lora_embedding_path', type=str, default="", help="path to the lora embedding model")
    parser.add_argument('--max_rank_num', type=int, default=0, help="the max rank for lora model")
    parser.add_argument('--max_embedding_rank_num', type=int, default=0, help="the max rank for lora embedding model")
    args = parser.parse_args()

    # load model
    origin_model, device, dtype = load_model(args)
    transformer = origin_model.model

    # convert_abnormal
    dir_path = "test_abnormal"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    convert_abnormal()
