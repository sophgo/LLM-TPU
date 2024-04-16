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
import datetime
import math
import unittest
import torch
import random
import sys
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM, LlamaConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='export Llama2 onnx.')
parser.add_argument('--model_path', type=str, default="../../torch2onnx/llama-2-13b-chat-hf", help='path to the torch model.')
parser.add_argument('--max_length', type=int, default=512, help="max sequence length")

args = parser.parse_args()

model_path = args.model_path
MAX_LEN = args.max_length
folder = "./tmp"

# origin_model = LlamaForCausalLM.from_pretrained(model_path)
origin_model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, device_map="auto").eval()
# origin_model.eval()
transformer = origin_model.model
config = origin_model.config

for param in origin_model.parameters():
    param.requires_grad = False

num_layers = config.num_hidden_layers
hidden_size = config.hidden_size
num_attention_heads = config.num_attention_heads
head_dim = hidden_size // num_attention_heads
layers = transformer.layers
tokenizer = LlamaTokenizer.from_pretrained(model_path)


class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        return transformer.embed_tokens(input_ids)


class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        # params
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids,
                                            use_cache=True)
        past_k, past_v = past_kv
        return hidden_states, past_k, past_v


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        # params
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids=position_ids,
                                            past_key_value=(past_k, past_v),
                                            use_cache=True)
        past_k, past_v = past_kv
        return hidden_states, past_k, past_v


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        _, token = torch.topk(m_logits, 1)
        return token


def convert_block(layer_id):
    # input
    # MAX_LEN + 1 for model combine
    hidden_states = torch.randn((1, MAX_LEN, hidden_size))
    position_ids = torch.tensor([range(MAX_LEN)], dtype=torch.long)
    attention_mask = -1000 * torch.ones((1, 1, MAX_LEN, MAX_LEN), dtype=torch.float32).triu(diagonal=1)
    model = Block(layer_id)
    # hiddeng_states = model(input_ids, position_ids)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'./tmp/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)


def convert_block_cache(layer_id):
    # input
    hidden_states = torch.randn((1, 1, hidden_size))
    position_ids = torch.tensor([range(1)], dtype=torch.long)
    attention_mask = -1000 * torch.ones((1, 1, 1, MAX_LEN + 1), dtype=torch.float32).triu(diagonal=0)
    past_k = torch.randn((1, MAX_LEN, num_attention_heads, head_dim))
    past_v = torch.randn((1, MAX_LEN, num_attention_heads, head_dim))
    model = BlockCache(layer_id)
    # hiddeng_states = model(input_ids, position_ids)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask, past_k, past_v),
        f'./tmp/block_cache_{layer_id}.onnx',
        verbose=False,
        input_names=[
            'input_states', 'position_ids', 'attention_mask', 'history_k',
            'history_v'
        ],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)


def convert_embedding():
    model = Embedding()
    input_ids = torch.tensor([range(MAX_LEN)])
    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f'{folder}/embedding.pt')

def convert_lm_head():
    model = LmHead()
    input = torch.randn(1, hidden_size)
    module = torch.jit.trace(model.forward, input)
    torch.jit.save(module, f'{folder}/lm_head.pt')

def test_net_with_mask():
    embed = Embedding()
    blocks = [Block(i) for i in range(num_layers)]
    block_kvs = [BlockCache(i) for i in range(num_layers)]
    ids = tokenizer.encode('hello')
    print("input ids:{}".format(ids))
    token_len = len(ids)
    ids = ids + (MAX_LEN - token_len) * [0]
    input_ids = torch.tensor(ids).view(MAX_LEN)
    out = embed(input_ids).view(1, MAX_LEN, hidden_size)
    position_ids = list(range(token_len)) + (MAX_LEN - token_len) * [0]
    position_ids = torch.tensor([position_ids])
    attention_mask = -1000 * torch.ones((MAX_LEN, MAX_LEN))
    for i in range(token_len):
        for j in range(token_len):
            if j <= i:
                attention_mask[i][j] = 0
    attention_mask = attention_mask.view(1, 1, MAX_LEN, MAX_LEN)
    k_cache = []
    v_cache = []
    for i in range(num_layers):
        out, k, v = blocks[i](out, position_ids, attention_mask)
        k[:,MAX_LEN - token_len:] = k[:,:token_len]
        v[:,MAX_LEN - token_len:] = v[:,:token_len]
        k[:,:MAX_LEN - token_len] = 0
        v[:,:MAX_LEN - token_len] = 0
        k_cache.append(k)
        v_cache.append(v)
    out = out[:,token_len - 1:token_len].view(1, hidden_size)
    lm = LmHead()
    token = lm(out).view(1)
    out_ids = [int(token)]
    word = tokenizer._convert_id_to_token(int(token[0]))
    print(word, end="")
    while token > 2 and token_len < 64:
        token_len += 1
        input_ids = torch.tensor([token])
        out = embed(input_ids).view(1, 1, hidden_size)
        position_ids = torch.tensor([[token_len - 1]])
        attention_mask = -1000 * torch.ones((1, 1, 1, MAX_LEN + 1))
        attention_mask[:, :, :, MAX_LEN + 1 - token_len:] = 0
        for i in range(num_layers):
            out, present_k_cache, present_v_cache = block_kvs[i](out, position_ids,
                                                    attention_mask,
                                                    k_cache[i], v_cache[i])
            new_k = torch.zeros(k_cache[i].shape)
            new_v = torch.zeros(v_cache[i].shape)
            new_k[:,MAX_LEN - token_len:MAX_LEN - 1] = k_cache[i][:,MAX_LEN - token_len + 1:]
            new_v[:,MAX_LEN - token_len:MAX_LEN - 1] = v_cache[i][:,MAX_LEN - token_len + 1:]
            new_k[:,MAX_LEN - 1:] = present_k_cache
            new_v[:,MAX_LEN - 1:] = present_v_cache
            k_cache[i] = new_k
            v_cache[i] = new_v
        token = lm(out).view(1)
        out_ids.append(int(token))
        word = tokenizer._convert_id_to_token(int(token[0]))
        print(word, end="")
    print("\noutput_ids:{}".format(out_ids))


#test_net_with_mask()

# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# export models
for i in range(num_layers):
    print("convert_block_{}".format(i))
    convert_block_cache(i)
    convert_block(i)
convert_embedding()
convert_lm_head()
