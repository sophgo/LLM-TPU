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
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='export onnx.')
parser.add_argument('--model_path', type=str, help='path to the torch model.')
parser.add_argument('--seq_length', type=int, default=512, help="sequence length")

args = parser.parse_args()

model_path = args.model_path
folder = f"./tmp/onnx"

device = torch.device("cpu")
origin_model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True,
    torch_dtype=torch.float).eval()

for param in origin_model.parameters():
    param.requires_grad = False

config = origin_model.config
transformer = origin_model.model
layers = transformer.layers

SEQ_LENGTH = args.seq_length
NUM_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS

print(f'Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n')

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        out = transformer.embed_tokens(input_ids)
        return out.float()


class QwenBlock(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
    # input_ids, attention_mask, position_ids
    def forward(self, hidden_states, position_ids, attention_mask):
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

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(
            hidden_states,
            past_key_value=(past_k, past_v),
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token
    
# refs:https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
class LmHeadTopk(torch.nn.Module):

    def __init__(self, top_k = 50, top_p = 0.8, min_tokens_to_keep = 5):
        super().__init__()
        self.top_k = top_k
        self.top_p = top_p
        self.min_tokens_to_keep = min_tokens_to_keep
        self.keep_matrix = torch.zeros((1, self.top_k), dtype=torch.bool)
        self.keep_matrix[0, :self.min_tokens_to_keep] = True

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        logits, token = torch.topk(m_logits.float(), self.top_k)

        # top_p
        cumulative_probs = logits.softmax(dim=1).cumsum(dim=1)
        mask = cumulative_probs < self.top_p
        mask = mask + self.keep_matrix
        filtered_logits = torch.where(mask, logits, torch.FloatTensor([-1000.]))
        probs = filtered_logits.softmax(dim=1)
        return probs, token

def convert_block(layer_id):
    model = QwenBlock(layer_id)
    hidden_states = torch.randn(
        (1, SEQ_LENGTH, HIDDEN_SIZE)).to(torch.float).to(device)
    position_ids = torch.tensor(
        [range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.randn(
        (1, 1, SEQ_LENGTH, SEQ_LENGTH)).to(torch.float).to(device)
    # import pdb;pdb.set_trace()
    # results = model(hidden_states, position_ids, attention_mask)
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
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).to(torch.float).to(device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device)
    attention_mask = torch.ones(
        (1, 1, 1, SEQ_LENGTH + 1)).to(torch.float).to(device)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(torch.float).to(device)
    past_v = torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(torch.float).to(device)

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
    input_ids = torch.tensor([range(SEQ_LENGTH)]).to(device)

    torch.onnx.export(model, (input_ids),
                      f'{folder}/embedding.onnx',
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['input_embed'],
                      do_constant_folding=True,
                      opset_version=15)


def convert_lm_head():
    model = LmHead()
    input = torch.randn(1, HIDDEN_SIZE).bfloat16().to(device)

    torch.onnx.export(model, (input),
                      f'{folder}/lm_head.onnx',
                      verbose=False,
                      input_names=['hidden_states'],
                      output_names=['token'],
                      do_constant_folding=True,
                      opset_version=15)


# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# export models
print(f'Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
    convert_block(i)
    convert_block_cache(i)

print(f'Convert embedding')
convert_embedding()

print(f'Convert lm_head')
convert_lm_head()

# if you want to generate outputs with topk
# print(f'Convert lm_head_topk')
# convert_lm_head()