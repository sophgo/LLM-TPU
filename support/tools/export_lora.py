#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import torch
import torch.nn as nn
import struct
import re
import argparse
import numpy as np
from transformers import AutoModelForCausalLM

torch.set_grad_enabled(False)


def convert_lora_to_bit(lora_model, lora_config, max_rank_num: int):
    # extract layer from model
    lora_weight_list = []
    is_regex = isinstance(lora_config.target_modules, str)
    target = lora_config.target_modules
    pattern = re.compile(target) if is_regex else None
    for i in range(len(lora_model.base_model.model.model.layers)):
        lora_layers = lora_model.base_model.model.model.layers[i]
        extracted_layers = {}

        for name, module in lora_layers.named_modules():
            if 'lora_A.default' not in name and 'lora_B.default' not in name:
                continue

            if not is_regex and any(t in name for t in target):
                extracted_layers[name] = module
            elif is_regex and pattern.search(name):
                extracted_layers[name] = module

        lora_A_weight_list = []
        lora_B_weight_list = []

        for name, extracted_layer in extracted_layers.items():
            lora_weight = extracted_layer.weight.detach().cpu().numpy().transpose(1, 0)
            left_dim, right_dim = lora_weight.shape

            if 'lora_A' in name and left_dim > right_dim:
                new_lora_weight = np.zeros((left_dim, max_rank_num), dtype=np.float32)
                new_lora_weight[:, :right_dim] = lora_weight
                lora_A_weight_list.append(new_lora_weight)
            elif 'lora_B' in name and left_dim < right_dim:
                new_lora_weight = np.zeros((max_rank_num, right_dim), dtype=np.float32)
                new_lora_weight[:left_dim, :] = lora_weight
                lora_B_weight_list.append(new_lora_weight)
            else:
                raise NotImplementedError

        # 由于在final.mlir中，weight的权重排列顺序是[lora_B, lora_A, lora_B, lora_A]的形式
        # 所以需要把B排列在前面
        for a, b in zip(lora_A_weight_list, lora_B_weight_list):
            lora_weight_list.append(a)
            lora_weight_list.append(b)

    # Flatten the weights and convert to uint32
    lora_weights_fp32 = np.concatenate([w.flatten() for w in lora_weight_list])
    lora_weights_fp32 = lora_weights_fp32
    lora_weights_uint32 = lora_weights_fp32.view(np.uint32)
    lora_weights_uint16 = (lora_weights_uint32 >> 16).astype(np.uint16)  # Convert to bfloat16

    if lora_weights_uint16.dtype.byteorder == '>':
        lora_weights_uint16 = lora_weights_uint16.byteswap()
    lora_weights_uint16 = lora_weights_uint16.newbyteorder('little')  # Ensure little-endian storage

    lora_weights_uint8_low = (lora_weights_uint16 >> 8).astype(np.uint8)
    lora_weights_uint8_high = (lora_weights_uint16 & 0xFF).astype(np.uint8)
    lora_weights_uint8 = np.column_stack(
        (lora_weights_uint8_high, lora_weights_uint8_low)).reshape(-1)

    return lora_weights_uint8


def convert_lora_embedding_to_bit(lora_model, max_embedding_rank_num: int):
    # extract layer from model
    lora_weight_list = []
    lora_layers = lora_model.base_model.model.model.embed_tokens
    extracted_layers = {}
    for name, module in lora_layers.named_modules():
        if 'lora_embedding_A' in name or 'lora_embedding_B' in name:
            extracted_layers[name] = module.default

    lora_A_weight_list = []
    lora_B_weight_list = []

    for name, extracted_layer in extracted_layers.items():
        lora_weight = extracted_layer.detach().cpu().numpy().transpose(1, 0)
        left_dim, right_dim = lora_weight.shape

        if 'lora_embedding_A' in name and left_dim > right_dim:
            new_lora_weight = np.zeros((left_dim, max_embedding_rank_num), dtype=np.float32)
            new_lora_weight[:, :right_dim] = lora_weight
            lora_A_weight_list.append(new_lora_weight)
        elif 'lora_embedding_B' in name and left_dim < right_dim:
            new_lora_weight = np.zeros((max_embedding_rank_num, right_dim), dtype=np.float32)
            new_lora_weight[:left_dim, :] = lora_weight
            lora_B_weight_list.append(new_lora_weight)
        else:
            raise NotImplementedError

    # 由于在final.mlir中，weight的权重排列顺序是[lora_B, lora_A]的形式
    # 但是在加载时，是按照算子调用逻辑来调用的，lora_A先走先调，lora_B后跑后调
    # 所以需要把A排列在前面
    for a, b in zip(lora_A_weight_list, lora_B_weight_list):
        lora_weight_list.append(a)
        lora_weight_list.append(b)

    # Flatten the weights and convert to uint32
    lora_weights_fp32 = np.concatenate([w.flatten() for w in lora_weight_list])
    lora_weights_uint32 = lora_weights_fp32.view(np.uint32)
    lora_weights_uint16 = (lora_weights_uint32 >> 16).astype(np.uint16)  # Convert to bfloat16

    if lora_weights_uint16.dtype.byteorder == '>':
        lora_weights_uint16 = lora_weights_uint16.byteswap()
    lora_weights_uint16 = lora_weights_uint16.newbyteorder('little')  # Ensure little-endian storage

    lora_weights_uint8_low = (lora_weights_uint16 >> 8).astype(np.uint8)
    lora_weights_uint8_high = (lora_weights_uint16 & 0xFF).astype(np.uint8)
    lora_weights_uint8 = np.column_stack(
        (lora_weights_uint8_high, lora_weights_uint8_low)).reshape(-1)

    return lora_weights_uint8


def make_header(size, header_size=64):
    if header_size < 8:
        raise ValueError("Header size must be at least 4 bytes to store the size.")
    header = np.zeros(header_size, dtype=np.uint8)
    size_bytes = struct.pack('<Q', header_size + size)
    header[:8] = np.frombuffer(size_bytes, dtype=np.uint8)
    return header


def convert_lora(args):
    from peft import PeftModel, PeftConfig
    # load model
    model_path = args.model_path
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 torch_dtype="auto",
                                                 device_map="auto",
                                                 attn_implementation="eager").eval()
    lora_config = PeftConfig.from_pretrained(args.lora_path)
    lora_model = PeftModel.from_pretrained(model, args.lora_path)

    lora_weights = convert_lora_to_bit(lora_model, lora_config, args.max_rank_num)
    if args.max_embedding_rank_num > 0:
        lora_embedding_weights = convert_lora_embedding_to_bit(lora_model,
                                                               args.max_embedding_rank_num)
        header = make_header(len(lora_weights) + len(lora_embedding_weights))
        total_lora_weights = np.concatenate([header, lora_weights, lora_embedding_weights])
    else:
        header = make_header(len(lora_weights))
        total_lora_weights = np.concatenate([header, lora_weights])
    # save and encrypt & decrypt
    with open(args.output, 'wb') as f:
        total_lora_weights.tofile(f)


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser(description='export lora as binary')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the torch model')
    parser.add_argument('-l', '--lora_path', type=str, required=True, help="path to the lora model")
    parser.add_argument('-o', '--output', type=str, required=True, help="output file name")
    parser.add_argument('--max_rank_num', type=int, default=32, help="the max rank for lora model")
    parser.add_argument('--max_embedding_rank_num', type=int, default=0, help="the max rank for lora embedding")
    args = parser.parse_args()
    # yapf: enable
    convert_lora(args)
