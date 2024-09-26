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
import json
import torch
import ctypes
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False)

class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        out = transformer.embed_tokens(input_ids)
        return out.float()


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


class BlockCache(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask, past_k, past_v):
        hidden_states, past_kv = self.layer(
            hidden_states,
            past_key_value=(past_k, past_v),
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            max_pos_len=args.max_pos_len
        )
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class LmHead(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        return m_logits


class GreedyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, m_logits):
        _, token = torch.topk(m_logits.float(), 1)
        return token


# refs:https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
class PenaltySampleHead(torch.nn.Module):
    def __init__(self, top_k=50, min_tokens_to_keep=5):
        super().__init__()
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep
        self.keep_matrix = torch.zeros((1, self.top_k), dtype=torch.bool)
        self.keep_matrix[0, : self.min_tokens_to_keep] = True

    def forward(self, m_logits, input_ids, top_p, temperature, penalty):
        # repeat penalty
        logits = torch.gather(m_logits, 1, input_ids)
        logits = torch.where(logits < 0, logits * penalty, logits / penalty)
        m_logits.scatter_(1, input_ids, logits)

        # top_k
        logits, token = torch.topk(m_logits.float(), self.top_k)

        # temperature
        logits = logits / temperature

        # top_p
        cumulative_probs = logits.softmax(dim=1).cumsum(dim=1)
        mask = cumulative_probs < top_p
        mask = mask + self.keep_matrix
        filtered_logits = torch.where(mask, logits, torch.FloatTensor([-1000.0]))
        probs = filtered_logits.softmax(dim=1)
        return probs, token


def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((1, SHARE_LENGTH, HIDDEN_SIZE)).to(device)
    position_ids = torch.tensor([range(SHARE_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.randn((1, 1, SHARE_LENGTH, SHARE_LENGTH)).to(device)

    torch.onnx.export(
        model,
        (hidden_states, position_ids, attention_mask),
        f"{folder}/block_{layer_id}.onnx",
        verbose=False,
        input_names=["input_states", "position_ids", "attention_mask"],
        output_names=["hidden_states", "past_k", "past_v"],
        do_constant_folding=True,
        opset_version=15,
    )


def convert_block_cache(layer_id):
    model = BlockCache(layer_id)
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device)
    attention_mask = torch.ones((1, 1, 1, SEQ_LENGTH + 1)).to(dtype).to(device)
    past_k = (
        torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)
    )
    past_v = (
        torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)
    )

    torch.onnx.export(
        model,
        (hidden_states, position_ids, attention_mask, past_k, past_v),
        f"{folder}/block_cache_{layer_id}.onnx",
        verbose=False,
        input_names=[
            "input_states",
            "position_ids",
            "attention_mask",
            "history_k",
            "history_v",
        ],
        output_names=["hidden_states", "past_k", "past_v"],
        do_constant_folding=True,
        opset_version=15,
    )


def convert_embedding():
    model = Embedding()
    input_ids = torch.tensor([range(SHARE_LENGTH)], dtype=torch.int32).to(device)
    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f"{folder}/embedding.pt")


def convert_lm_head():
    model = LmHead()
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).to(dtype).to(device)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f"{folder}/lm_head.pt")


def convert_greedy_head():
    model = GreedyHead()
    m_logits = torch.randn(1, VOCAB_SIZE)

    torch.onnx.export(
        model,
        (m_logits),
        f"{folder}/greedy_head.onnx",
        verbose=False,
        input_names=["m_logits"],
        output_names=["token"],
        do_constant_folding=True,
        opset_version=15,
    )


def convert_penalty_sample_head():
    model = PenaltySampleHead()
    m_logits = torch.randn(1, VOCAB_SIZE)
    input_ids = torch.tensor([range(SEQ_LENGTH)])
    top_p = torch.tensor([0.8])
    temperature = torch.tensor([0.98])
    penalty = torch.tensor([0.98])

    torch.onnx.export(
        model,
        (m_logits, input_ids, top_p, temperature, penalty),
        f"{folder}/penalty_sample_head.onnx",
        verbose=False,
        input_names=["m_logits", "input_ids", "top_p", "temperature", "penalty"],
        output_names=["probs", "token"],
        do_constant_folding=True,
        opset_version=15,
    )

def fp32_string(data):
    return bin(ctypes.c_uint32.from_buffer(ctypes.c_float(data)).value)[2:]

def convert_embedding_to_bit():
    print("\033[31m请注意！！如果embedding_mode=binary，目前convert_embedding_to_bit只支持embedding为float32格式，并且导出格式为bfloat16！！！\033[0m")
    print("\033[31m如果想导出float16的embedding，请修改此函数！！！\033[0m")
    embedding_weights = transformer.embed_tokens.weight.data
    embedding_weights_fp32 = embedding_weights.numpy().astype(np.float32).flatten()
    embedding_weights_uint32 = embedding_weights_fp32.view(np.uint32)
    embedding_weights_uint16 = (embedding_weights_uint32 >> 16).astype(np.uint16) # torch的格式必须是bfloat16才行
    if embedding_weights_uint16.dtype.byteorder == '>':
        embedding_weights_uint16 = embedding_weights_uint16.byteswap()
    embedding_weights_uint16 = embedding_weights_uint16.newbyteorder('little') # 确保数据以小端序存储

    with open('embedding.bin', 'wb') as f:
        embedding_weights_uint16.tofile(f)


def convert_lora_to_bit():
    from peft import LoraConfig, PeftModel
    # 1. load lora
    config_file = os.path.join(args.lora_path, "adapter_config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Neither config.json nor adapter_config.json found in {args.lora_path}")
    with open(config_file) as f:
        lora_config_dict = json.load(f)
    lora_config = LoraConfig(**lora_config_dict)
    lora_model = PeftModel.from_pretrained(origin_model, args.lora_path)

    # 2. extract layer from model
    lora_weight_list = []
    for i in range(NUM_LAYERS):
        lora_layers = lora_model.base_model.model.model.layers[i]
        extracted_layers = {}

        for name, module in lora_layers.named_modules():
            if 'lora_A.default' in name or 'lora_B.default' in name:
                if any(layer_name in name for layer_name in list(lora_config.target_modules)):
                    extracted_layers[name] = module

        lora_A_weight_list = []
        lora_B_weight_list = []

        for name, extracted_layer in extracted_layers.items():
            lora_weight = extracted_layer.weight.detach().cpu().numpy().transpose(0,1)
            left_dim, right_dim = lora_weight.shape

            if 'lora_A' in name:
                new_lora_weight = np.zeros((args.max_rank_num, right_dim), dtype=np.float32)
                new_lora_weight[:left_dim, :] = lora_weight
                lora_A_weight_list.append(new_lora_weight)
            elif 'lora_B' in name:
                new_lora_weight = np.zeros((left_dim, args.max_rank_num), dtype=np.float32)
                new_lora_weight[:, :right_dim] = lora_weight
                lora_B_weight_list.append(new_lora_weight)

        # 由于在final.mlir中，weight的权重排列顺序是[lora_B, lora_A, lora_B, lora_A]的形式
        # 所以需要把B排列在前面
        for a, b in zip(lora_A_weight_list, lora_B_weight_list):
            lora_weight_list.append(b)
            lora_weight_list.append(a)

    # Flatten the weights and convert to uint32
    lora_weights_fp32 = np.concatenate([w.flatten() for w in lora_weight_list])
    lora_weights_uint32 = lora_weights_fp32.view(np.uint32)
    lora_weights_uint16 = (lora_weights_uint32 >> 16).astype(np.uint16)  # Convert to bfloat16

    if lora_weights_uint16.dtype.byteorder == '>':
        lora_weights_uint16 = lora_weights_uint16.byteswap()
    lora_weights_uint16 = lora_weights_uint16.newbyteorder('little')  # Ensure little-endian storage

    with open('lora_weights.bin', 'wb') as f:
        lora_weights_uint16.tofile(f)

def setup_environment():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return

def load_model():
    # setup environment
    setup_environment()

    # set
    device = torch.device(args.device)
    if args.device == "cpu":
        dtype = torch.float
        torch.set_num_threads(args.num_threads)
    else:
        dtype = torch.bfloat16

    # load model
    model_path = args.model_path
    origin_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto"
    ).eval()
    for param in origin_model.parameters():
        param.requires_grad = False
    return origin_model, device, dtype

def convert():
    # create folder to store onnx
    if not os.path.exists(folder):
        os.makedirs(folder)

    # export lora model
    print("Convert lora")
    convert_lora_to_bit()

    # export models
    print("Convert block & block_cache")
    for i in tqdm(range(NUM_LAYERS)):
        convert_block(i)
        convert_block_cache(i)

    print('Convert embedding')
    if args.embedding_mode == "default":
        convert_embedding()
    elif args.embedding_mode == "binary":
        convert_embedding_to_bit()

    print("Convert lm_head")
    convert_lm_head()
    convert_greedy_head()
    convert_penalty_sample_head()

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='export onnx')
    parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
    parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-s', '--seq_length', type=int, default=512, help="sequence length")
    parser.add_argument('-n', '--num_threads', type=int, default=1, help='The number of threads used for torch if device is cpu')
    parser.add_argument('--share_length', type=int, default=6144, help="share length")
    parser.add_argument('--unshare_length', type=int, default=4096, help="unshare length")
    parser.add_argument('--max_pos_len', type=int, default=8704, help="max position length")
    parser.add_argument('--generation_mode', type=str, default="default", choices=["default", "lmhead_with_penalty", "lmhead_with_sample", "lmhead_with_top1"], help="generation mode")
    parser.add_argument('--embedding_mode', type=str, default="default", choices=["default", "binary"], help="if set embedding_mode=binary, will save embedding.bin and infer without tpu")
    parser.add_argument('--lora_path', type=str, default="", help="path to the lora model")
    parser.add_argument('--max_rank_num', type=int, default=0, help="the max rank for lora model")
    args = parser.parse_args()

    # load model
    origin_model, device, dtype = load_model()
    config = origin_model.config
    transformer = origin_model.model
    layers = transformer.layers
    SEQ_LENGTH = args.seq_length
    SHARE_LENGTH = args.share_length
    UNSHARE_LENGTH = args.unshare_length
    BATCH_SIZE = args.batch_size
    NUM_LAYERS = config.num_hidden_layers
    HIDDEN_SIZE = config.hidden_size
    NUM_ATTENTION_HEADS = config.num_attention_heads
    NUM_KEY_VALUE_HEADS = config.num_key_value_heads
    HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
    VOCAB_SIZE = config.vocab_size
    print(f"Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n")
    folder = f"./tmp_share{args.share_length}_unshare{args.unshare_length}_seq{args.seq_length}/onnx"

    # convert
    convert()
