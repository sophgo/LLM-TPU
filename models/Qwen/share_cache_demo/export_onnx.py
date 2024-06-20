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
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
parser.add_argument('-b', '--batch_size', type=int, default=2, help='batch size')
parser.add_argument('-s', '--seq_length', type=int, default=512, help="sequence length")
parser.add_argument('-n', '--num_threads', type=int, default=1, help='The number of threads used for torch if device is cpu')
parser.add_argument('--share_length', type=int, default=6144, help="share length")
parser.add_argument('--unshare_length', type=int, default=4096, help="unshare length")

args = parser.parse_args()

def modify_json(json_path):
    with open(json_path, 'r') as file:
        config_json = json.load(file)
    # config_json['seq_length'] = args.seq_length
    config_json['fp16'] = False
    config_json['bf16'] = False
    config_json['fp32'] = True
    config_json['use_dynamic_ntk'] = False
    with open(json_path, 'w') as file:
        json.dump(config_json, file, indent=4)

model_path = args.model_path
json_path = os.path.join(model_path, "config.json")
folder = f"./tmp/onnx"

device = torch.device(args.device)
if device == 'cpu':
    modify_json(json_path) # warning!!!!!
    torch.set_num_threads(args.num_threads)

origin_model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True,
    torch_dtype=torch.float, device_map="auto").eval()

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


class QwenBlock(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        # self.rotary_emb = transformer.rotary_emb(SEQ_LENGTH)
        # self.cos_emb = self.rotary_emb[0].view(SEQ_LENGTH, HEAD_DIM)
        # self.sin_emb = self.rotary_emb[1].view(SEQ_LENGTH, HEAD_DIM)

    def forward(self, hidden_states, position_ids, attention_mask):
        # cos_pos = self.cos_emb[position_ids].unsqueeze(2)
        # sin_pos = self.sin_emb[position_ids].unsqueeze(2)
        hidden_states, past_kv = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class QwenBlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        # self.rotary_emb = transformer.rotary_emb(SEQ_LENGTH)
        # self.cos_emb = self.rotary_emb[0].view(SEQ_LENGTH, HEAD_DIM)
        # self.sin_emb = self.rotary_emb[1].view(SEQ_LENGTH, HEAD_DIM)

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        # cos_pos = self.cos_emb[position_ids].unsqueeze(2)
        # sin_pos = self.sin_emb[position_ids].unsqueeze(2)
        hidden_states, past_kv = self.layer(
            hidden_states,
            layer_past=(past_k, past_v),
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()

class QwenBlockShareCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = transformer.rotary_emb(SEQ_LENGTH)
        self.cos_emb = self.rotary_emb[0].view(SEQ_LENGTH, HEAD_DIM)
        self.sin_emb = self.rotary_emb[1].view(SEQ_LENGTH, HEAD_DIM)

    def forward(self, hidden_states, position_ids, share_attention_mask, unshare_attention_mask, 
                share_past_k, share_past_v, unshare_past_k, unshare_past_v):
        cos_pos = self.cos_emb[position_ids].unsqueeze(2)
        sin_pos = self.sin_emb[position_ids].unsqueeze(2)
        hidden_states, past_kv = self.layer(
            hidden_states,
            layer_past=(share_past_k, share_past_v, unshare_past_k, unshare_past_v),
            attention_mask=(share_attention_mask, unshare_attention_mask),
            rotary_pos_emb_list=[[cos_pos, sin_pos]],
            use_cache=True)
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
    

def convert_block(layer_id):
    model = QwenBlock(layer_id)
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
    model = QwenBlockCache(layer_id)
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
    model = QwenBlockCache(layer_id)
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


def convert_block_share_cache(layer_id):
    model = QwenBlockShareCache(layer_id)
    hidden_states = torch.randn((BATCH_SIZE, 1, HIDDEN_SIZE)).to(device)
    position_ids = torch.tensor(BATCH_SIZE * [range(1)], dtype=torch.long).to(device)
    share_attention_mask = torch.ones((1, 1, 1, SHARE_LENGTH)).to(device)
    unshare_attention_mask = torch.ones((BATCH_SIZE, 1, 1, UNSHARE_LENGTH)).to(device)
    share_past_k = torch.randn((1, SHARE_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(device)
    share_past_v = torch.randn((1, SHARE_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(device)    
    unshare_past_k = torch.randn((BATCH_SIZE, UNSHARE_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(device)
    unshare_past_v = torch.randn((BATCH_SIZE, UNSHARE_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(device)

    torch.onnx.export(
        model, (hidden_states, position_ids, share_attention_mask, unshare_attention_mask, share_past_k, share_past_v, unshare_past_k, unshare_past_v),
        f'{folder}/block_share_cache_{layer_id}.onnx',
        verbose=False,
        input_names=[
            'input_states', 'position_ids', 'share_attention_mask', 'unshare_attention_mask', 'share_past_k',
            'share_past_v', 'unshare_past_k', 'unshare_past_v'
        ],
        output_names=['hidden_states', 'unshare_present_k', 'unshare_present_v'],
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

def test_net_with_mask():
    import numpy as np
    block_cache = QwenBlockCache(0)
    block_share_cache = QwenBlockShareCache(0)
    hidden_states = torch.randn((BATCH_SIZE, 1, HIDDEN_SIZE)).to(device)
    position_ids = torch.tensor(BATCH_SIZE * [range(1)], dtype=torch.long).to(device)
    share_attention_mask = torch.zeros((1, 1, 1, SHARE_LENGTH)).to(device)
    unshare_attention_mask = torch.zeros((BATCH_SIZE, 1, 1, UNSHARE_LENGTH)).to(device)
    share_past_k = torch.randn((1, SHARE_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(device)
    share_past_v = torch.randn((1, SHARE_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(device)    
    unshare_past_k = torch.randn((BATCH_SIZE, UNSHARE_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(device)
    unshare_past_v = torch.randn((BATCH_SIZE, UNSHARE_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(device)
    output, unshare_present_k, unshare_present_v = block_share_cache(hidden_states, position_ids, share_attention_mask, unshare_attention_mask, share_past_k, share_past_v, unshare_past_k, unshare_past_v)

    attention_mask = torch.zeros((1, 1, 1, SHARE_LENGTH + UNSHARE_LENGTH + 1)).to(device)
    past_k = torch.cat([share_past_k, unshare_past_k[:1]], dim=1)
    past_v = torch.cat([share_past_v, unshare_past_v[:1]], dim=1)
    output_cach, present_k, present_v = block_cache(hidden_states[:1], position_ids[:1], attention_mask, past_k, past_v)


# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# test_net_with_mask()

# export models
print(f'Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
    convert_block(i) # prefill
    convert_block_unshare(i)
    convert_block_cache(i) # decode

print(f'Convert embedding')
convert_embedding()

print(f'Convert lm_head')
convert_lm_head()
convert_greedy_head()
convert_penalty_sample_head()

print("Done")

