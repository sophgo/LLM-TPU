#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
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

class Block(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            max_pos_len=args.max_pos_len
        )
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()

class DequantBlock(torch.nn.Module):
    def __init__(self, layer_id, dequant_weight_dic, op_name_list):
        super().__init__()
        self.layer_id = layer_id
        __layers = copy.deepcopy(layers)
        self.layer = __layers[layer_id]

        # assign
        self.layer.self_attn.q_proj.weight = dequant_weight_dic[op_name_list[0]]
        self.layer.self_attn.k_proj.weight = dequant_weight_dic[op_name_list[1]]
        self.layer.self_attn.v_proj.weight = dequant_weight_dic[op_name_list[2]]
        self.layer.self_attn.o_proj.weight = dequant_weight_dic[op_name_list[3]]
        
        self.layer.mlp.gate_proj.weight = dequant_weight_dic[op_name_list[4]]
        self.layer.mlp.up_proj.weight = dequant_weight_dic[op_name_list[5]]
        self.layer.mlp.down_proj.weight = dequant_weight_dic[op_name_list[6]]

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            max_pos_len=args.max_pos_len
        )
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


def dequant(npz_file, op_name, q_group_size, hidden_size):
    weight = npz_file[op_name + "_uint"]
    scale = npz_file[op_name + "_reordered"]
    zp = npz_file[op_name + "_1_reordered"]
    scale = uint16_to_float32(scale.flatten()).reshape(q_group_size,-1,hidden_size // q_group_size).transpose(1,0,2).flatten().round(6)
    zp = zp.reshape(q_group_size,-1,hidden_size // q_group_size).transpose(1,0,2).flatten()
    weight_high = weight & 0x0F
    weight_low = weight >> 4
    
    scale = np.repeat(scale, q_group_size // 2)
    zp = np.repeat(zp, q_group_size // 2)
    
    dequant_weights_high = (weight_high.astype(np.int32) - zp) * scale
    dequant_weights_low = (weight_low.astype(np.int32) - zp) * scale

    dequant_weight = np.empty(dequant_weights_high.size + dequant_weights_low.size, dtype=np.float32)
    dequant_weight[0::2] = dequant_weights_high
    dequant_weight[1::2] = dequant_weights_low
    return dequant_weight

def get_dequant_weight_dic(fp32_file, npz_file, fp32_op_name_list, op_name_list, op_shape_list, q_group_size, cos_sim_threshold, verify=True):
    dequant_weight_dic = {}
    for fp32_op_name, op_name, op_shape in zip(fp32_op_name_list, op_name_list, op_shape_list):
        dequant_weight = dequant(npz_file, op_name, q_group_size, op_shape[1]) # 这里用op_shape[1]而不是HIDDEN_SIZE

        if verify:
            fp32_weight = fp32_file[fp32_op_name].flatten()
            dequant_bf16_weight = dequant_weight.reshape(op_shape).transpose(1,0).flatten()
            cos_sim = cosine_similarity(fp32_weight, dequant_bf16_weight)
            if cos_sim < cos_sim_threshold:
                raise ValueError(f"cos_sim : {cos_sim}, failed")
        dequant_torch_weight = torch.FloatTensor(dequant_weight.reshape(op_shape))
        dequant_weight_dic[op_name] = torch.nn.Parameter(dequant_torch_weight, requires_grad=False)
    return dequant_weight_dic

def exec_cmd(i, dir_path, tpu_in_pcie):
    os.system(f'cd {dir_path} && \
        model_transform.py \
            --model_name block_{i} \
            --model_def block_{i}.onnx \
            --mlir block_{i}.mlir \
            --test_input test_input.npz \
            --test_result mlir_outputs.npz && \
        model_deploy.py \
            --mlir block_{i}.mlir \
            --quantize W4BF16 \
            --chip bm1684x \
            --model block_{i}.bmodel \
            --quant_input \
            --quant_output \
            --addr_mode io_alone \
            --debug')
    if tpu_in_pcie:
        os.system(f'model_runner.py \
            --input block_{i}_in_f32.npz \
            --model block_{i}_bm1684x_w4bf16_tpu.mlir \
            --output block_{i}_bm1684x_w4bf16_tpu_outputs.npz && \
        model_runner.py \
            --input block_{i}_in_f32.npz \
            --model block_{i}.bmodel \
            --output block_{i}_bm1684x_w4bf16_model_outputs.npz')
    os.system("cd ..")


def test_block(tpu_in_pcie):
    dir_path = "test_block"
    q_group_size = 64
    cos_sim_threshold = 0.97
    setup_environment()
    # create folder to store onnx
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    input_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE)).to(device)
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.randn((1, 1, SEQ_LENGTH, SEQ_LENGTH)).to(device)
    inputs = {
        'input_states': input_states.numpy(),
        'position_ids': position_ids.numpy(),
        'attention_mask': attention_mask.numpy()}
    np.savez(f"{dir_path}/test_input.npz", **inputs)

    for i in tqdm(range(NUM_LAYERS)):
        block = Block(i)

        torch.onnx.export(
            block,
            (input_states, position_ids, attention_mask),
            f"{dir_path}/block_{i}.onnx",
            verbose=False,
            input_names=["input_states", "position_ids", "attention_mask"],
            output_names=["hidden_states", "past_k", "past_v"],
            do_constant_folding=True,
            opset_version=15,
        )

        exec_cmd(i, dir_path, tpu_in_pcie)

        fp32_npz_name = f"{dir_path}/block_{i}_top_f32_all_weight.npz"
        addressed_npz_name = f"{dir_path}/block_{i}_tpu_addressed_bm1684x_w4bf16_weight.npz"
        fp32_file = np.load(fp32_npz_name)
        npz_file = np.load(addressed_npz_name)

        dequant_weight_dic = get_dequant_weight_dic(fp32_file, npz_file, fp32_op_name_list, op_name_list, op_shape_list, q_group_size, cos_sim_threshold)
        dequant_block = DequantBlock(i, dequant_weight_dic, op_name_list)

        output = block(input_states, position_ids, attention_mask)
        dequant_output = dequant_block(input_states, position_ids, attention_mask)
        for output_i, dequant_output_i in zip(output, dequant_output):
            cos_sim_0 = cosine_similarity(output_i.numpy().flatten(), dequant_output_i.numpy().flatten()) # torch & dequant torch
            print(f"fp32的torch结果 与 反量化回torch的结果，余弦相似度为：{cos_sim_0}")

        if tpu_in_pcie:
            for dequant_output_i, bmodel_output_name_i in zip(dequant_output, bmodel_output.files):
                cos_sim_1 = cosine_similarity(dequant_output_i.numpy().flatten(), bmodel_output[bmodel_output_name_i].flatten()) # dequant torch & w4bf16 bmodel
                print(f"bmodel结果 与 反量化回torch的结果，余弦相似度为：{cos_sim_1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='export onnx')
    parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
    parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-s', '--seq_length', type=int, default=512, help="sequence length")
    parser.add_argument('-n', '--num_threads', type=int, default=1, help='The number of threads used for torch if device is cpu')
    parser.add_argument('--prefill_length', type=int, default=6144, help="prefill length")
    parser.add_argument('--max_pos_len', type=int, default=8704, help="max position length")
    parser.add_argument('--tpu_in_pcie', action='store_true', help="when exists tpu in pcie, please set")
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

    test_block(args.tpu_in_pcie)
    print("Block验证完毕")
