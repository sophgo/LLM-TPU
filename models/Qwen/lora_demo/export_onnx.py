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
import json
import torch
import ctypes
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False)

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

args = parser.parse_args()

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

model_path = args.model_path
folder = f"./tmp_share{args.share_length}_unshare{args.unshare_length}_seq{args.seq_length}/onnx"

device = torch.device(args.device)
if args.device == "cpu":
    dtype = torch.float
    torch.set_num_threads(args.num_threads)
else:
    dtype = torch.bfloat16

origin_model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto"
).eval()

for param in origin_model.parameters():
    param.requires_grad = False

config = origin_model.config
transformer = origin_model.transformer
layers = transformer.h


SEQ_LENGTH = args.seq_length
SHARE_LENGTH = args.share_length
UNSHARE_LENGTH = args.unshare_length
BATCH_SIZE = args.batch_size
NUM_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
VOCAB_SIZE = config.vocab_size

print(f'Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n')

class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        out = transformer.wte(input_ids)
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
            max_pos_len=args.max_pos_len)
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class BlockCache(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(
            hidden_states,
            layer_past=(past_k, past_v),
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            max_pos_len=args.max_pos_len)
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class LmHeadWithTopK(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.ln_f(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token

class LmHead(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.ln_f(hidden_states)
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
    def __init__(self, top_k = 50, min_tokens_to_keep = 5):
        super().__init__()
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep
        self.keep_matrix = torch.zeros((1, self.top_k), dtype=torch.bool)
        self.keep_matrix[0, :self.min_tokens_to_keep] = True

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
        filtered_logits = torch.where(mask, logits, torch.FloatTensor([-1000.]))
        probs = filtered_logits.softmax(dim=1)
        return probs, token


class LmHeadWithTop1Head(torch.nn.Module):

    def __init__(self, top_k = 50, min_tokens_to_keep = 5):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.ln_f(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token


class LmHeadWithSampleHead(torch.nn.Module):

    def __init__(self, top_k = 50, min_tokens_to_keep = 5):
        super().__init__()
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep
        self.keep_matrix = torch.zeros((1, self.top_k), dtype=torch.bool)
        self.keep_matrix[0, :self.min_tokens_to_keep] = True

    def forward(self, hidden_states, top_p, temperature):
        hidden_states = transformer.ln_f(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)

        # top_k
        logits, token = torch.topk(m_logits.float(), self.top_k)

        # temperature
        logits = logits / temperature

        # top_p
        cumulative_probs = logits.softmax(dim=1).cumsum(dim=1)
        mask = cumulative_probs < top_p
        mask = mask + self.keep_matrix
        filtered_logits = torch.where(mask, logits, torch.FloatTensor([-1000.]))
        probs = filtered_logits.softmax(dim=1)
        return probs, token


class LmHeadWithPenaltySampleHead(torch.nn.Module):

    def __init__(self, top_k = 50, min_tokens_to_keep = 5):
        super().__init__()
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep
        self.keep_matrix = torch.zeros((1, self.top_k), dtype=torch.bool)
        self.keep_matrix[0, :self.min_tokens_to_keep] = True

    def forward(self, hidden_states, input_ids, top_p, temperature, penalty):
        hidden_states = transformer.ln_f(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)

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
        filtered_logits = torch.where(mask, logits, torch.FloatTensor([-1000.]))
        probs = filtered_logits.softmax(dim=1)
        return probs, token


def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn(
        (1, SHARE_LENGTH, HIDDEN_SIZE)).to(device)
    position_ids = torch.tensor(
        [range(SHARE_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.randn(
        (1, 1, SHARE_LENGTH, SHARE_LENGTH)).to(device)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)


def convert_block_cache(layer_id):
    model = BlockCache(layer_id)
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).to(device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device)
    attention_mask = torch.ones(
        (1, 1, 1, SEQ_LENGTH + 1)).to(device)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(device)
    past_v = torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(device)

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


def convert_block_unshare(layer_id):
    model = BlockCache(layer_id)
    hidden_states = torch.randn((1, UNSHARE_LENGTH, HIDDEN_SIZE)).to(device)
    position_ids = torch.tensor([range(UNSHARE_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.ones(
        (1, 1, UNSHARE_LENGTH, SHARE_LENGTH + UNSHARE_LENGTH)).to(device)
    past_k = torch.randn((1, SHARE_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(device)
    past_v = torch.randn((1, SHARE_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(device)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask, past_k, past_v),
        f'{folder}/block_unshare_{layer_id}.onnx',
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
    input_ids = torch.tensor([range(SHARE_LENGTH)], dtype=torch.int32).to(device)
    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f'{folder}/embedding.pt')


def convert_lm_head_with_topk():
    model = LmHeadWithTopK()
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).to(device)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head_with_topk.pt')

def convert_lm_head():
    model = LmHead()
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).to(device)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head.pt')


def convert_greedy_head():
    model = GreedyHead()
    m_logits = torch.randn(1, VOCAB_SIZE)

    torch.onnx.export(
        model, (m_logits),
        f'{folder}/greedy_head.onnx',
        verbose=False,
        input_names=['m_logits'],
        output_names=['token'],
        do_constant_folding=True,
        opset_version=15)


def convert_penalty_sample_head():
    model = PenaltySampleHead()
    m_logits = torch.randn(1, VOCAB_SIZE)
    input_ids = torch.tensor([range(SEQ_LENGTH)])
    top_p = torch.tensor([0.8])
    temperature = torch.tensor([0.98])
    penalty = torch.tensor([0.98])

    torch.onnx.export(
        model, (m_logits, input_ids, top_p, temperature, penalty),
        f'{folder}/penalty_sample_head.onnx',
        verbose=False,
        input_names=[
            'm_logits', 'input_ids', 'top_p', 'temperature',
            'penalty'
        ],
        output_names=['probs', 'token'],
        do_constant_folding=True,
        opset_version=15)
    

def convert_lmhead_with_top1_head():
    model = LmHeadWithTop1Head()

    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).to(device)
    module = torch.jit.trace(model.forward, (hidden_states))
    torch.jit.save(module, f'{folder}/lm_head.pt')


def convert_lmhead_with_sample_head():
    model = LmHeadWithSampleHead()

    hidden_states = torch.randn(1, HIDDEN_SIZE).to(device)
    top_p = torch.tensor([0.8])
    temperature = torch.tensor([0.98])
    module = torch.jit.trace(model.forward, (hidden_states, top_p, temperature))
    torch.jit.save(module, f'{folder}/lm_head.pt')


def convert_lmhead_with_penalty_sample_head():
    model = LmHeadWithPenaltySampleHead()

    hidden_states = torch.randn(1, HIDDEN_SIZE).to(device)
    input_ids = torch.tensor([range(SEQ_LENGTH)])
    top_p = torch.tensor([0.8])
    temperature = torch.tensor([0.98])
    penalty = torch.tensor([0.98])
    module = torch.jit.trace(model.forward, (hidden_states, input_ids, top_p, temperature, penalty))
    torch.jit.save(module, f'{folder}/lm_head.pt')

def fp32_string(data):
    return bin(ctypes.c_uint32.from_buffer(ctypes.c_float(data)).value)[2:]

def convert_embedding_to_bit():
    print("\033[31m请注意！！如果embedding_mode=binary，目前convert_embedding_to_bit只支持embedding为float32格式，并且导出格式为bfloat16！！！\033[0m")
    print("\033[31m如果想导出float16的embedding，请修改此函数！！！\033[0m")
    embedding_weights = transformer.wte.weight.data
    embedding_weights_fp32 = embedding_weights.numpy().astype(np.float32).flatten()
    embedding_weights_uint32 = embedding_weights_fp32.view(np.uint32)
    embedding_weights_uint16 = (embedding_weights_uint32 >> 16).astype(np.uint16) # torch的格式必须是bfloat16才行
    if embedding_weights_uint16.dtype.byteorder == '>':
        embedding_weights_uint16 = embedding_weights_uint16.byteswap()
    embedding_weights_uint16 = embedding_weights_uint16.newbyteorder('little') # 确保数据以小端序存储

    with open('embedding.bin', 'wb') as f:
        embedding_weights_uint16.tofile(f)


# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# test_net_with_mask()

# export models
print('Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
    convert_block(i) # prefill
    if args.unshare_length!=0:
        convert_block_unshare(i)
    convert_block_cache(i) # decode

print('Convert embedding')
if args.embedding_mode == "default":
    convert_embedding()
elif args.embedding_mode == "binary":
    convert_embedding_to_bit()

print('Convert lm_head')
if args.generation_mode == "default":
    convert_lm_head()
    convert_greedy_head()
    convert_penalty_sample_head()
elif args.generation_mode == "lmhead_with_penalty":
    convert_lmhead_with_penalty_sample_head()
elif args.generation_mode == "lmhead_with_sample":
    convert_lmhead_with_sample_head()
elif args.generation_mode == "lmhead_with_top1":
    convert_lmhead_with_top1_head()

print("Done")

