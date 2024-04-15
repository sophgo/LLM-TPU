#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

# pip install transformers_stream_generator einops tiktoken
# export PYTHONPATH=$PWD/../../Qwen-7B-Chat:$PYTHONPATH
import datetime
import math
import unittest
import torch
import random
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
parser.add_argument('-n', '--num_threads', type=int, default=1, help='The number of threads used for torch if device is cpu')

args = parser.parse_args()

QWEN_PATH = args.model_path
device = torch.device(args.device)
torch.set_num_threads(args.num_threads)

folder = "./tmp/onnx"

origin_model = AutoModelForCausalLM.from_pretrained(
    QWEN_PATH, trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
transformer = origin_model.transformer
layers = transformer.h

NUM_LAYERS = len(layers)
SEQ_LENGTH = transformer.seq_length
HIDDEN_SIZE = layers[0].attn.hidden_size
NUM_HEADS = layers[0].attn.num_heads
for param in origin_model.parameters():
    param.requires_grad = False


class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        out = transformer.wte(input_ids)
        return out.float()


class QwenBlock(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        # params
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = transformer.rotary_emb(SEQ_LENGTH)
        self.cos_emb = self.rotary_emb[0].view(SEQ_LENGTH, 128)
        self.sin_emb = self.rotary_emb[1].view(SEQ_LENGTH, 128)

    def forward(self, hidden_states, position_ids, attention_mask):
        cos_pos = self.cos_emb[position_ids].unsqueeze(2)
        sin_pos = self.sin_emb[position_ids].unsqueeze(2)
        hidden_states, past_kv = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb_list=[[cos_pos, sin_pos]],
            use_cache=True)
        past_k, past_v = past_kv
        return hidden_states.float(), past_k.float(), past_v.float()


class QwenBlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        # params
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = transformer.rotary_emb(SEQ_LENGTH)
        self.cos_emb = self.rotary_emb[0].view(SEQ_LENGTH, 128)
        self.sin_emb = self.rotary_emb[1].view(SEQ_LENGTH, 128)

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        cos_pos = self.cos_emb[position_ids].unsqueeze(2)
        sin_pos = self.sin_emb[position_ids].unsqueeze(2)
        hidden_states, past_kv = self.layer(
            hidden_states,
            layer_past=(past_k, past_v),
            attention_mask=attention_mask,
            rotary_pos_emb_list=[[cos_pos, sin_pos]],
            use_cache=True)
        k, v = past_kv
        return hidden_states.float(), k.float(), v.float()


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.ln_f(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token


def convert_qwen_block(layer_id):
    # input
    hidden_states = torch.randn(
        (1, SEQ_LENGTH, HIDDEN_SIZE)).bfloat16().to(device)
    position_ids = torch.tensor(
        [range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.randn(
        (1, 1, SEQ_LENGTH, SEQ_LENGTH)).bfloat16().to(device)
    model = QwenBlock(layer_id)
    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)


def convert_qwen_block_cache(layer_id):
    # input
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).bfloat16().to(device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device)
    attention_mask = torch.ones(
        (1, 1, 1, SEQ_LENGTH + 1)).bfloat16().to(device)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_HEADS, 128)).bfloat16().to(device)
    past_v = torch.randn((1, SEQ_LENGTH, NUM_HEADS, 128)).bfloat16().to(device)
    model = QwenBlockCache(layer_id)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask, past_k, past_v),
        f'{folder}/block_cache_{layer_id}.onnx',
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
    input = torch.tensor([range(SEQ_LENGTH)], dtype=torch.int32).to(device)
    module = torch.jit.trace(model.forward, input)
    torch.jit.save(module, f'{folder}/embedding.pt')

def convert_lm_head():
    model = LmHead()
    input = torch.randn(1, HIDDEN_SIZE).bfloat16().to(device)
    module = torch.jit.trace(model.forward, input)
    torch.jit.save(module, f'{folder}/lm_head.pt')

def build_prompt(query):
    return f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'

# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# export models
print(f'Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
    convert_qwen_block_cache(i)
    convert_qwen_block(i)

print("convert_embedding")
convert_embedding()

print("convert_lm_head")
convert_lm_head()
