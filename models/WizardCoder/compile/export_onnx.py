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
folder = f'./tmp/onnx'

origin_model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True).eval()

for param in origin_model.parameters():
    param.requires_grad = False

config = origin_model.config
transformer = origin_model.transformer

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
        self.wte = transformer.wte
        self.wpe = transformer.wpe

    def forward(self, input_ids, position_ids):
        return self.wte(input_ids) + self.wpe(position_ids)


class Block(torch.nn.Module):

    def __init__(self, layer_id) -> None:
        super().__init__()
        self.layer = transformer.h[layer_id]

    def forward(self, hidden_states, attention_mask=None):
        hidden_states, past_layer = self.layer(hidden_states, use_cache=True,
                                               attention_mask=attention_mask)
        return hidden_states, past_layer


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id) -> None:
        super().__init__()
        self.layer = transformer.h[layer_id]

    def forward(self, hidden_states, layer_past, attention_mask=None):
        hidden_states, past_layer = self.layer(hidden_states, layer_past,
                                               use_cache=True, attention_mask=attention_mask)
        return hidden_states, past_layer


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ln = transformer.ln_f
        self.lm_head = origin_model.lm_head

    def forward(self, hidden_states):
        x = self.ln(hidden_states)
        logits = self.lm_head(x)
        _, token = torch.topk(logits, 1)
        return token


def convert_block(layer_id):
    model = Block(layer_id).eval()
    hidden_states = torch.rand((1, SEQ_LENGTH, HIDDEN_SIZE))
    attention_mask = -1000 * torch.ones((1, 1, SEQ_LENGTH, SEQ_LENGTH), dtype=torch.float32).triu(diagonal=1)

    torch.onnx.export(
        model, (hidden_states, attention_mask),
        f'{folder}/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'attention_mask'],
        output_names=['hidden_states', 'past_layer'],
        do_constant_folding=True,
        opset_version=15)


def convert_block_cache(layer_id):
    model = BlockCache(layer_id).eval()
    hidden_states = torch.rand((1, 1, HIDDEN_SIZE))
    past_layer = torch.rand((1, SEQ_LENGTH, HEAD_DIM * 2))
    attention_mask = -1000 * torch.ones((1, 1, 1, SEQ_LENGTH + 1), dtype=torch.float32).triu(diagonal=1)
    
    torch.onnx.export(
        model, (hidden_states, past_layer, attention_mask),
        f'{folder}/block_cache_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'past_cache', 'attention_mask'],
        output_names=['hidden_states', 'current_cache'],
        do_constant_folding=True,
        opset_version=15)


def convert_embedding():
    model = Embedding()
    input_ids = torch.tensor([range(SEQ_LENGTH)])
    position_ids = torch.tensor([range(SEQ_LENGTH)])

    torch.onnx.export(model, (input_ids, position_ids),
                      f'{folder}/embedding.onnx',
                      verbose=False,
                      input_names=['input_ids', 'input_pos'],
                      output_names=['hidden_state'],
                      do_constant_folding=True,
                      opset_version=15)


def convert_lm_head():
    model = LmHead()
    input = torch.randn(1, HIDDEN_SIZE)

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
