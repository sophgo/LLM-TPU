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
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('--model_path', type=str, default ="./Phi-3-mini/" ,help='path to the torch model')
parser.add_argument('--seq_length', type=int, default=512, help="sequence length")

args = parser.parse_args()

model_path = args.model_path
folder = f"./tmp/onnx"

origin_model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True).eval()

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
VOCAB_SIZE = config.vocab_size
print(f'Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n')

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        hidden_states = transformer.embed_tokens(input_ids)
        return hidden_states


class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask):
        # breakpoint()
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids,
                                            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        # breakpoint()
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids=position_ids,
                                            past_key_value=(past_k, past_v),
                                            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v


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
    model = Block(layer_id)
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE))
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long)
    attention_mask = -1000 * torch.ones((1, 1, SEQ_LENGTH, SEQ_LENGTH), dtype=torch.float32).triu(diagonal=1)
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
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE))
    position_ids = torch.tensor([range(1)], dtype=torch.long)
    attention_mask = -1000 * torch.ones((1, 1, 1, SEQ_LENGTH + 1), dtype=torch.float32).triu(diagonal=1)
    past_k = torch.randn((1, SEQ_LENGTH, config.num_key_value_heads, HEAD_DIM))
    past_v = torch.randn((1, SEQ_LENGTH, config.num_key_value_heads, HEAD_DIM))
    results = model(hidden_states, position_ids, attention_mask, past_k, past_v)
    
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
    input_ids = torch.tensor([range(SEQ_LENGTH)])

    torch.onnx.export(model, (input_ids),
                      f'{folder}/embedding.onnx',
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['input_embed'],
                      do_constant_folding=True,
                      opset_version=15)


def convert_lm_head():
    model = LmHead()
    input = torch.randn(1, HIDDEN_SIZE)

    torch.onnx.export(model, (input),
                      f'{folder}/lm_head.onnx',
                      verbose=False,
                      input_names=['hidden_states'],
                      output_names=['m_logits'],
                      do_constant_folding=True,
                      opset_version=15)

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

# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# export models
print(f'Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
    convert_block_cache(i)
    convert_block(i)

print(f'Convert embedding')
convert_embedding()

print(f'Convert lm_head')
convert_lm_head()
convert_greedy_head()
convert_penalty_sample_head()
print("Done")

# def test_net_with_mask():
#     import numpy as np
#     num_layers = NUM_LAYERS
#     MAX_LEN = SEQ_LENGTH
#     embed = Embedding()
#     blocks = [Block(i) for i in range(num_layers)]
#     block_kvs = [BlockCache(i) for i in range(num_layers)]
#     ids = tokenizer.encode('你好')
#     query = '你好'
#     print(query)
#     # promt = tokenizer.build_prompt(query)
#     # ids = tokenizer.encode(promt)
#     ids = [    1, 32006,   887,   526,   263,  8444, 13436, 20255, 29889,  3529,
#           3867,  9109, 29892, 11314,   936,   322, 16232,  2472,   304,   278,
#           1404, 29889, 32007, 32010, 11644,   526,   366, 29973, 32007, 32001]
#     MAX_LEN = len(ids) + 20
#     print("input ids:{}".format(ids))
#     token_len = len(ids)
#     ids = ids + (MAX_LEN - token_len) * [0]
#     input_ids = torch.tensor(ids).view(MAX_LEN)

#     out = embed(input_ids).view(1, MAX_LEN, HIDDEN_SIZE)
#     # breakpoint()
#     position_ids = list(range(token_len)) + (MAX_LEN - token_len) * [0]
#     position_ids = torch.tensor([position_ids])
#     attention_mask = torch.ones((MAX_LEN, MAX_LEN)) * -10000.0
#     for i in range(token_len):
#         attention_mask[i,:i+1] = 0
#     attention_mask = attention_mask.view((1,1,MAX_LEN,MAX_LEN))
#     k_cache = []
#     v_cache = []

#     for i in tqdm(range(num_layers)):
#         out, k, v = blocks[i](out, position_ids, attention_mask)
#         k[:,token_len:,:,:] = 0
#         v[:,token_len:,:,:] = 0
#         k_cache.append(k)
#         v_cache.append(v)
#     out = out[0, token_len - 1:token_len, :].view(1, HIDDEN_SIZE)
#     lm = LmHead()
#     m_logits = lm(out)
#     greedy_head = GreedyHead()
#     token = greedy_head(m_logits)
#     out_ids = [int(token)]
#     word = tokenizer._convert_id_to_token(int(token[0]))
#     print(word, end="")
#     for i in tqdm(range(10)):
#         token_len += 1
#         input_ids = torch.tensor([token])
#         out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
#         position_ids = torch.tensor([[token_len - 1]])
#         attention_mask = torch.ones((1, 1, 1, MAX_LEN + 1)) * -10000.0
#         attention_mask[:, :, :, :token_len] = 0
#         attention_mask[:, :, :, -1] = 0

#         for i in range(num_layers):
#             out, k_cache_present, v_cache_present = block_kvs[i](out, position_ids,
#                                                        attention_mask,
#                                                        k_cache[i], v_cache[i])
#             k_cache[i][:,token_len-1:token_len,:,:] = k_cache_present
#             v_cache[i][:,token_len-1:token_len,:,:] = v_cache_present
#         m_logits = lm(out)
#         token = greedy_head(m_logits)
#         out_ids.append(int(token))
#         word = tokenizer._convert_id_to_token(int(token[0]))
#         print(word, end="")
#     print("\noutput_ids:{}".format(out_ids))

# test_net_with_mask()

#My: [306, 626, 385, 319, 29902, 13436, 20255, 8688, 304, 3867, 8444]
#    [306, 626, 385, 319, 29902, 13436, 20255, 8688, 304,  3867, 8444, 29892,  9109, 29892,   322, 16232,  2472,   304,  4160, 29889, 1619,  6437,   338,   304,  6985,   366,   411,   263,  9377,  3464]