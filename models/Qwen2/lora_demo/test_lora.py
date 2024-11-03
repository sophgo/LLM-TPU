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
import copy
import json
import torch
import torch.nn as nn
import ctypes
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)

from export_onnx import load_model, load_lora_model, setup_environment, convert_total_lora_to_bit
from test_a16matmul import uint16_to_float32, cosine_similarity
from test_block import get_dequant_weight_dic

def test_lora(lora_scale, lora_offset):
    folder = f"./test_block"
    q_group_size = 64
    cos_sim_threshold = 0.98

    lora_model, lora_config = load_lora_model(origin_model, args.lora_path)
    for i in range(NUM_LAYERS):
        # hook dequant weight from npz in compile
        fp32_npz_name = f"{folder}/block_{i}_top_f32_all_weight.npz"
        addressed_npz_name = f"{folder}/block_{i}_tpu_addressed_bm1684x_w4bf16_weight.npz"
        fp32_file = np.load(fp32_npz_name)
        npz_file = np.load(addressed_npz_name)
        dequant_weight_dic = get_dequant_weight_dic(fp32_file, npz_file, fp32_op_name_list, op_name_list, op_shape_list, q_group_size, cos_sim_threshold)

        # assign dequant weight to model
        cur_layer = lora_model.base_model.model.model.layers[i]
        cur_layer.self_attn.q_proj.base_layer.weight = dequant_weight_dic[op_name_list[0]]
        cur_layer.self_attn.k_proj.base_layer.weight = dequant_weight_dic[op_name_list[1]]
        cur_layer.self_attn.v_proj.base_layer.weight = dequant_weight_dic[op_name_list[2]]
        cur_layer.self_attn.o_proj.base_layer.weight = dequant_weight_dic[op_name_list[3]]

        cur_layer.mlp.gate_proj.base_layer.weight = dequant_weight_dic[op_name_list[4]]
        cur_layer.mlp.up_proj.base_layer.weight = dequant_weight_dic[op_name_list[5]]
        cur_layer.mlp.down_proj.base_layer.weight = dequant_weight_dic[op_name_list[6]]

        # assign lora weight to model
        for name, module in cur_layer.named_modules():
            if 'lora_A.default' in name or 'lora_B.default' in name:
                if any(layer_name in name for layer_name in list(lora_config.target_modules)):
                    module.weight = torch.nn.Parameter((module.weight + lora_offset) * lora_scale, requires_grad=False)

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = lora_model.generate(
        model_inputs.input_ids,
        max_new_tokens=20
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

def test_lora_embedding(lora_embedding_scale, lora_offset):
    folder = f"./test_block"
    q_group_size = 64
    cos_sim_threshold = 0.98

    lora_model, lora_config = load_lora_model(origin_model, args.lora_embedding_path)
    lora_embed = lora_model.base_model.model.model.embed_tokens
    for name, module in lora_embed.named_modules():
        if 'lora_embedding_A' in name or 'lora_embedding_B' in name:
            module.default = torch.nn.Parameter((module.default + lora_offset) * lora_embedding_scale, requires_grad=False)

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = lora_model.generate(
        model_inputs.input_ids,
        max_new_tokens=20
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='export onnx')
    parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
    parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-s', '--seq_length', type=int, default=512, help="sequence length")
    parser.add_argument('-n', '--num_threads', type=int, default=1, help='The number of threads used for torch if device is cpu')
    parser.add_argument('--prefill_length', type=int, default=6144, help="prefill length")
    parser.add_argument('--max_pos_len', type=int, default=8704, help="max position length")
    parser.add_argument('--generation_mode', type=str, default="default", choices=["default", "lmhead_with_penalty", "lmhead_with_sample", "lmhead_with_top1"], help="generation mode")
    parser.add_argument('--lib_path', type=str, default='', help='lib path by user')
    parser.add_argument('--lora_path', type=str, default="", help="path to the lora model")
    parser.add_argument('--lora_embedding_path', type=str, default="", help="path to the lora embedding model")
    parser.add_argument('--max_rank_num', type=int, default=0, help="the max rank for lora model")
    parser.add_argument('--max_embedding_rank_num', type=int, default=0, help="the max rank for lora embedding model")
    args = parser.parse_args()

    # load model
    origin_model, device, dtype = load_model(args)
    config = origin_model.config
    transformer = origin_model.model
    layers = transformer.layers
    SEQ_LENGTH = args.seq_length
    SHARE_LENGTH = args.prefill_length
    BATCH_SIZE = args.batch_size
    NUM_LAYERS = config.num_hidden_layers
    HIDDEN_SIZE = config.hidden_size
    INTERMEDIATE_SIZE = config.intermediate_size
    NUM_ATTENTION_HEADS = config.num_attention_heads
    NUM_KEY_VALUE_HEADS = config.num_key_value_heads
    HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
    VOCAB_SIZE = config.vocab_size
    print(f"Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # parameters
    fp32_op_name_list = [
        "onnx::MatMul_233",
        "onnx::MatMul_234",
        "onnx::MatMul_235",
        "onnx::MatMul_281",
        "onnx::MatMul_282",
        "onnx::MatMul_283",
        "onnx::MatMul_284"
    ]
    op_name_list = [
        "/layer/self_attn/q_proj/Add_output_0_Add",
        "/layer/self_attn/k_proj/Add_output_0_Add",
        "/layer/self_attn/v_proj/Add_output_0_Add",
        "/layer/self_attn/o_proj/MatMul_output_0_MatMul",
        "/layer/mlp/gate_proj/MatMul_output_0_MatMul",
        "/layer/mlp/up_proj/MatMul_output_0_MatMul",
        "/layer/mlp/down_proj/MatMul_output_0_MatMul",
    ]
    op_shape_list = [
        [HIDDEN_SIZE, HIDDEN_SIZE],
        [512, HIDDEN_SIZE],
        [512, HIDDEN_SIZE],
        [HIDDEN_SIZE, HIDDEN_SIZE],
        [INTERMEDIATE_SIZE, HIDDEN_SIZE],
        [INTERMEDIATE_SIZE, HIDDEN_SIZE],
        [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    ]

    dir_path = "test_lora"
    convert_total_lora_to_bit(f"{dir_path}/encrypted_lora_weights_0_0.bin", origin_model, 0, 0, 0, args)
    convert_total_lora_to_bit(f"{dir_path}/encrypted_lora_weights_1_0.bin", origin_model, 1, 0, 0, args)
    convert_total_lora_to_bit(f"{dir_path}/encrypted_lora_weights_0_1.bin", origin_model, 0, 1, 0, args)
    convert_total_lora_to_bit(f"{dir_path}/encrypted_lora_weights_10_0.bin", origin_model, 10, 0, 0.0001, args)
    convert_total_lora_to_bit(f"{dir_path}/encrypted_lora_weights_20_0.bin", origin_model, 20, 0, 0.0001, args)
    convert_total_lora_to_bit(f"{dir_path}/encrypted_lora_weights_30_0.bin", origin_model, 30, 0, 0.0001, args)
    convert_total_lora_to_bit(f"{dir_path}/encrypted_lora_weights_0_10.bin", origin_model, 0, 10, 0.0001, args)
    convert_total_lora_to_bit(f"{dir_path}/encrypted_lora_weights_0_20.bin", origin_model, 0, 20, 0.0001, args)
    convert_total_lora_to_bit(f"{dir_path}/encrypted_lora_weights_0_30.bin", origin_model, 0, 30, 0.0001, args)

    print("-------------------test_lora-------------------")
    test_lora(lora_scale=1, lora_offset=0.0)
    test_lora(lora_scale=10, lora_offset=0.0001)
    test_lora(lora_scale=20, lora_offset=0.0001)
    test_lora(lora_scale=30, lora_offset=0.0001)

    print("-------------------test_lora_embedding-------------------")
    test_lora_embedding(lora_embedding_scale=1, lora_offset=0.0)
    test_lora_embedding(lora_embedding_scale=10, lora_offset=0.0001)
    test_lora_embedding(lora_embedding_scale=20, lora_offset=0.0001)
    test_lora_embedding(lora_embedding_scale=30, lora_offset=0.0001)
