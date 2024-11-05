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

from export_onnx import load_model, load_lora_model, setup_environment
from test_a16matmul import uint16_to_float32, cosine_similarity
from test_block import get_dequant_weight_dic

def test_llm():
    dir_path = "test_llm"
    folder = f"./test_block"
    setup_environment()
    # create folder to store onnx
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cos_sim_threshold = 0.98
    q_group_size = 64
    for i in range(NUM_LAYERS):
        # hook dequant weight from npz in compile
        fp32_npz_name = f"{folder}/block_{i}_top_f32_all_weight.npz"
        addressed_npz_name = f"{folder}/block_{i}_tpu_addressed_bm1684x_w4bf16_weight.npz"
        fp32_file = np.load(fp32_npz_name)
        npz_file = np.load(addressed_npz_name)
        dequant_weight_dic = get_dequant_weight_dic(fp32_file, npz_file, fp32_op_name_list, op_name_list, op_shape_list, q_group_size, cos_sim_threshold)

        # assign dequant weight to model
        cur_layer = origin_model.model.layers[i]
        cur_layer.self_attn.q_proj.weight = dequant_weight_dic[op_name_list[0]]
        cur_layer.self_attn.k_proj.weight = dequant_weight_dic[op_name_list[1]]
        cur_layer.self_attn.v_proj.weight = dequant_weight_dic[op_name_list[2]]
        cur_layer.self_attn.o_proj.weight = dequant_weight_dic[op_name_list[3]]

        cur_layer.mlp.gate_proj.weight = dequant_weight_dic[op_name_list[4]]
        cur_layer.mlp.up_proj.weight = dequant_weight_dic[op_name_list[5]]
        cur_layer.mlp.down_proj.weight = dequant_weight_dic[op_name_list[6]]

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
    generated_ids = origin_model.generate(
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

    test_llm()
    print("反量化回torch的全流程 验证完毕")
