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

def uint16_to_float32(uint16_array, mode = "bf16"):
    return (uint16_array.astype(np.uint32) << 16).view(np.float32)

def cosine_similarity(matrix1, matrix2):
    if isinstance(matrix1, list):
        matrix1 = np.array(matrix1, dtype=np.float32)
    if isinstance(matrix2, list):
        matrix2 = np.array(matrix2, dtype=np.float32)

    assert matrix1.shape == matrix2.shape, "Matrices must have the same shape."
    dot_product = np.dot(matrix1.flatten(), matrix2.flatten())
    norm_matrix1 = np.linalg.norm(matrix1)
    norm_matrix2 = np.linalg.norm(matrix2)
    similarity = dot_product / (norm_matrix1 * norm_matrix2)
    return similarity

class NetA(torch.nn.Module):

    def __init__(self):
        super(NetA, self).__init__()
        self.filter = torch.randn((512, 128))
        self.act = torch.nn.SiLU()

    def forward(self, x):
        a = torch.matmul(x, self.filter)
        b = self.act(a)
        return b

class NetB(torch.nn.Module):

    def __init__(self, weight):
        super(NetB, self).__init__()
        self.filter = torch.FloatTensor(weight.reshape(512,128))
        self.act = torch.nn.SiLU()

    def forward(self, x):
        a = torch.matmul(x, self.filter)
        b = self.act(a)
        return b

class NetC(torch.nn.Module):

    def __init__(self, weight, rank):
        super(NetC, self).__init__()
        self.filter = torch.FloatTensor(weight.reshape(512,128))
        self.act = torch.nn.SiLU()

        self.lora_A = torch.zeros(512, rank)
        self.lora_B = torch.zeros(rank, 128)

    def forward(self, x):
        a = torch.matmul(x, self.filter)
        lora_out = torch.matmul(torch.matmul(x, self.lora_A), self.lora_B)
        b = self.act(a + lora_out)
        return b

def exec_cmd(dir_path):
    os.system(f'cd {dir_path} && \
        model_transform.py \
            --model_name test_a \
            --model_def test_a.onnx \
            --mlir test_a.mlir \
            --test_input test_input.npz \
            --test_result mlir_outputs.npz && \
        model_deploy.py \
            --mlir test_a.mlir \
            --quantize W4BF16 \
            --chip bm1684x \
            --model test_a.bmodel \
            --quant_input \
            --quant_output \
            --addr_mode io_alone \
            --debug \
            --test_input test_input.npz \
            --test_reference mlir_outputs.npz && \
        cd ..')

def test_a16matmul():
    dir_path = "test_a16matmul"
    q_group_size = 64
    cos_sim_threshold = 0.99
    setup_environment()
    # create folder to store onnx
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    x = torch.randn(4, 512).float()

    inputs = {'x': x.numpy()}
    np.savez(f"{dir_path}/test_input.npz", **inputs)

    net_a = NetA()
    torch.onnx.export(net_a, (x),
                    f"{dir_path}/test_a.onnx",
                    export_params=True,
                    verbose=True,
                    opset_version=13,
                    input_names=['x'])

    exec_cmd(dir_path)

    npz_file = np.load(f"{dir_path}/test_a_tpu_addressed_bm1684x_w4bf16_weight.npz")
    weight = npz_file['/MatMul_output_0_MatMul_uint']
    scale = npz_file['/MatMul_output_0_MatMul_reordered']
    zp = npz_file['/MatMul_output_0_MatMul_1_reordered']
    new_weight = []

    scale = uint16_to_float32(scale.flatten()).reshape(64,2,8).transpose(1,0,2).flatten().round(6)
    zp = zp.reshape(64,2,8).transpose(1,0,2).flatten()
    weight_high = weight & 0x0F
    weight_low = weight >> 4

    for i in range(0, len(weight) * 2, 2):
        quant_idx = i // q_group_size
        scale_i = scale[quant_idx]
        zp_i = zp[quant_idx]
        new_weight.append(((int(weight_high[i // 2]) - zp_i) * scale_i))
        new_weight.append(((int(weight_low[i // 2]) - zp_i) * scale_i))

    fp32_weight = np.load(f"{dir_path}/test_a_top_f32_all_weight.npz")["/Constant_output_0"].flatten()
    dequant_bf16_weight = np.array(new_weight, dtype=np.float32).reshape(128,512).transpose(1,0).flatten()
    cos_sim = cosine_similarity(fp32_weight, dequant_bf16_weight)

    net_b = NetB(dequant_bf16_weight)
    net_c = NetC(dequant_bf16_weight, 64)

    output_a = net_a(x)
    output_b = net_b(x)
    output_c = net_c(x)

    output_bmodel = np.load(f"{dir_path}/test_a_bm1684x_w4bf16_model_outputs.npz")["4_Mul"]
    cos_sim = cosine_similarity(output_c.numpy().flatten(), output_bmodel.flatten())

    print(f"bmodel结果 与 反量化回torch的结果，余弦相似度为：{cos_sim}")


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

    test_a16matmul()
    print("A16MatMul算子验证完毕")
