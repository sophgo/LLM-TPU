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
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False)

from export_onnx import load_model, setup_environment
from export_onnx import Embedding, BlockCache, LmHead, GreedyHead, PenaltySampleHead

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

def test_matmul():
    setup_environment()
    # create folder to store onnx
    if not os.path.exists("test_onnx"):
        os.makedirs("test_onnx")

    torch.manual_seed(0)
    x = torch.randn(4, 512).float()

    inputs = {'x': x.numpy()}
    np.savez("test_onnx/test_input.npz", **inputs)

    net_a = NetA()
    torch.onnx.export(net_a, (x),
                    "test_onnx/test_a.onnx",
                    export_params=True,
                    verbose=True,
                    opset_version=13,
                    input_names=['x'])

    npz_file = np.load("test_onnx/test_a_tpu_addressed_bm1684x_w4bf16_weight.npz")
    weight = npz_file['/MatMul_output_0_MatMul_uint']
    scale = npz_file['/MatMul_output_0_MatMul_reordered']
    zp = npz_file['/MatMul_output_0_MatMul_1_reordered']
    q_group_size = 64
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

    fp32_weight = np.load("test_onnx/test_a_top_f32_all_weight.npz")["/Constant_output_0"].flatten()
    dequant_bf16_weight = np.array(new_weight, dtype=np.float32).reshape(128,512).transpose(1,0).flatten()
    cos_sim = cosine_similarity(fp32_weight, dequant_bf16_weight)

    net_b = NetB(dequant_bf16_weight)
    net_c = NetC(dequant_bf16_weight, 64)

    output_a = net_a(x)
    output_b = net_b(x)
    output_c = net_c(x)

    output_bmodel = np.load("test_onnx/test_a_bm1684x_w4bf16_model_outputs.npz")["4_Mul"]
    cos_sim = cosine_similarity(output_c.numpy().flatten(), output_bmodel.flatten())

    # 对比bmodel结果与pytorch结果，且lora都置零 （一个int4 matmul）
    if cos_sim < 0.99:
        raise ValueError(f"cos_sim : {cos_sim}, failed")

def build_prompt(query):
    return f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'


def test_net_with_mask():
    embed = Embedding().to(device)
    blocks = [Block(i).to(device) for i in range(NUM_LAYERS)]
    block_kvs = [BlockCache(i).to(device) for i in range(NUM_LAYERS)]
    lm = LmHead()
    greedy_head = GreedyHead()
    query = """tell me about sophgo in ten word"""
    print(query)
    promt = build_prompt(query)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ids = tokenizer.encode(promt)
    print("input ids:{}".format(ids))

    token_len = len(ids)
    ori_token_len = token_len
    ids = ids + (SEQ_LENGTH - token_len) * [0]
    input_ids = torch.tensor(ids).view(SEQ_LENGTH).to(device)
    out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)
    position_ids = list(range(token_len)) + (SEQ_LENGTH - token_len) * [0]
    position_ids = torch.tensor([position_ids]).to(device)
    attention_mask = torch.ones((SEQ_LENGTH, SEQ_LENGTH)).float() * -10000.0
    for i in range(token_len):
        for j in range(token_len):
            if j <= i:
                attention_mask[i][j] = 0.0
    attention_mask = attention_mask.view(1, 1, SEQ_LENGTH, SEQ_LENGTH).to(device)
    k_cache = []
    v_cache = []
    for i in range(NUM_LAYERS):
        out[:,token_len:] = 0
        out, k, v = blocks[i](out.to(dtype),
                              position_ids,
                              attention_mask)
        k_cache.append(k)
        v_cache.append(v)
    out = out[:, token_len - 1:token_len].view(1, 1, HIDDEN_SIZE)
    token = greedy_head(lm(out.to(dtype))).view(1)
    out_ids = [int(token)]
    word = tokenizer.decode([int(token)])
    print(word, end="")
    while int(token) != tokenizer.eos_token_id and token_len <= ori_token_len + 10:
        token_len += 1
        input_ids = torch.tensor([token]).to(device)
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]]).to(device)
        attention_mask = torch.zeros((1, 1, 1, SEQ_LENGTH + 1)).float().to(device)
        attention_mask[:, :, :, token_len:SEQ_LENGTH] = -10000.0
        for i in range(NUM_LAYERS):
            out, k, v = block_kvs[i](out.to(dtype),
                                     position_ids,
                                     attention_mask,
                                     k_cache[i].to(dtype),
                                     v_cache[i].to(dtype))
            k_cache[i][:, token_len:token_len+1] = k
            v_cache[i][:, token_len:token_len+1] = v
        token = greedy_head(lm(out.to(dtype))).view(1)
        out_ids.append(int(token))
        word = tokenizer.decode([int(token)])
        print(word, end="")
    print("\noutput_ids:{}".format(out_ids))

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

def test_llm():
    dir_path = "test_llm"
    folder = "tmp_prefill1024_seq1024/int4_1dev/block/"
    setup_environment()
    # create folder to store onnx
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # parameters
    cos_sim_threshold = 0.98
    q_group_size = 64
    fp32_op_name_list = [
        "onnx::MatMul_237",
        "onnx::MatMul_238",
        "onnx::MatMul_239",
        "onnx::MatMul_285",
        "onnx::MatMul_286",
        "onnx::MatMul_287",
        "onnx::MatMul_288"
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
    input_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE)).to(device)
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.randn((1, 1, SEQ_LENGTH, SEQ_LENGTH)).to(device)
    inputs = {
        'input_states': input_states.numpy(),
        'position_ids': position_ids.numpy(),
        'attention_mask': attention_mask.numpy()}
    np.savez("test_llm/test_input.npz", **inputs)

    for i in tqdm(range(NUM_LAYERS)):
        fp32_npz_name = f"{folder}/block_cache_{i}_top_f32_all_weight.npz"
        addressed_npz_name = f"{folder}/block_cache_{i}_tpu_addressed_bm1684x_w4bf16_weight.npz"
        fp32_file = np.load(fp32_npz_name)
        npz_file = np.load(addressed_npz_name)

        dequant_weight_dic = {}
        for fp32_op_name, op_name, op_shape in zip(fp32_op_name_list, op_name_list, op_shape_list):
            dequant_weight = dequant(npz_file, op_name, q_group_size, op_shape[1]) # 这里用op_shape[1]而不是HIDDEN_SIZE
            fp32_weight = fp32_file[fp32_op_name].flatten()
            dequant_bf16_weight = dequant_weight.reshape(op_shape).transpose(1,0).flatten()
            cos_sim = cosine_similarity(fp32_weight, dequant_bf16_weight)
            if cos_sim < cos_sim_threshold:
                raise ValueError(f"cos_sim : {cos_sim}, failed")
            dequant_torch_weight = torch.FloatTensor(dequant_weight.reshape(op_shape))
            dequant_weight_dic[op_name] = torch.nn.Parameter(dequant_torch_weight, requires_grad=False)

        block = Block(i)
        dequant_block = DequantBlock(i, dequant_weight_dic, op_name_list)

        output = block(input_states, position_ids, attention_mask)
        dequant_output = dequant_block(input_states, position_ids, attention_mask)
        bmodel_output = np.load(f"{folder}/block_{i}_bm1684x_w4bf16_model_outputs.npz")
        for output_i, dequant_output_i, bmodel_output_name_i in zip(output, dequant_output, bmodel_output.files):
            cos_sim_0 = cosine_similarity(output_i.numpy().flatten(), dequant_output_i.numpy().flatten()) # torch & dequant torch
            cos_sim_1 = cosine_similarity(dequant_output_i.numpy().flatten(), bmodel_output[bmodel_output_name_i].flatten()) # dequant torch & w4bf16 bmodel
            if cos_sim_0 < cos_sim_threshold or cos_sim_1 < cos_sim_threshold:
                raise ValueError(f"cos_sim_0 : {cos_sim_0}, cos_sim_1 : {cos_sim_1}, failed")


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
    parser.add_argument('--embedding_mode', type=str, default="default", choices=["default", "binary"], help="if set embedding_mode=binary, will save embedding.bin and infer without tpu")
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

    # convert
    # convert()
    # test_matmul()
    test_llm()
    # test_net_with_mask()
