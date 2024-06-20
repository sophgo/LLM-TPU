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
from tqdm import tqdm
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
parser.add_argument('-s', '--seq_length', type=int, default=512, help="sequence length")
parser.add_argument('-n', '--num_threads', type=int, default=1, help='The number of threads used for torch if device is cpu')
parser.add_argument('--share_length', type=int, default=6144, help="share length")
parser.add_argument('--unshare_length', type=int, default=4096, help="unshare length")

args = parser.parse_args()

#def modify_json(json_path):
#    with open(json_path, 'r') as file:
#        config_json = json.load(file)
#    config_json['seq_length'] = args.seq_length
#    with open(json_path, 'w') as file:
#        json.dump(config_json, file, indent=4)

model_path = args.model_path
json_path = os.path.join(model_path, "config.json")
folder = f"./tmp/onnx"

#modify_json(json_path)

device = torch.device(args.device)
origin_model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True,
    torch_dtype=torch.float).eval()

for param in origin_model.parameters():
    param.requires_grad = False

config = origin_model.config
transformer = origin_model.model
layers = transformer.layers

SEQ_LENGTH = args.seq_length
SHARE_LENGTH = args.share_length
UNSHARE_LENGTH = args.unshare_length
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
            use_cache=True)
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
    hidden_states = torch.randn(
        (1, SHARE_LENGTH, HIDDEN_SIZE)).to(torch.float).to(device)
    position_ids = torch.tensor(
        [range(SHARE_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.randn(
        (1, 1, SHARE_LENGTH, SHARE_LENGTH)).to(torch.float).to(device)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
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


def convert_block_cache(layer_id):
    model = BlockCache(layer_id)
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
    input_ids = torch.tensor([range(SHARE_LENGTH)], dtype=torch.int32).to(device)
    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f'{folder}/embedding.pt')
    

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

def build_prompt(query):
    return f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{query}'

def test_net_with_mask():
    embed = Embedding()
    blocks = [Block(i) for i in range(NUM_LAYERS)]
    blocks_unshare = [BlockCache(i) for i in range(NUM_LAYERS)]
    block_kvs = [BlockCache(i) for i in range(NUM_LAYERS)]
    query = """"""
    print(query)
    promt = build_prompt(query)
    import numpy as np
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ids = tokenizer.encode(promt)
    print("input ids:{}".format(ids))
    token_len = len(ids)
    ori_token_len = token_len
    ids = ids + (SHARE_LENGTH - token_len) * [0]
    input_ids = torch.tensor(ids).view(SHARE_LENGTH).to(device)
    out = embed(input_ids).view(1, SHARE_LENGTH, HIDDEN_SIZE)
    position_ids = list(range(token_len)) + (SHARE_LENGTH - token_len) * [0]
    position_ids = torch.tensor([position_ids]).to(device)
    attention_mask = torch.ones((SHARE_LENGTH, SHARE_LENGTH)).float() * -10000.0
    for i in range(token_len):
        for j in range(token_len):
            if j <= i:
                attention_mask[i][j] = 0.0
    attention_mask = attention_mask.view(1, 1, SHARE_LENGTH, SHARE_LENGTH).to(device)
    k_cache = torch.zeros(NUM_LAYERS, 1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)
    v_cache = torch.zeros(NUM_LAYERS, 1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)
    for i in range(NUM_LAYERS):
        out, k, v = blocks[i](out, position_ids, attention_mask)
        k_cache[i][:, :token_len]= k[:, :token_len]
        v_cache[i][:, :token_len]= v[:, :token_len]

    # unshare
    query2 = """tell me about sophgo in ten word<|im_end|>\n<|im_start|>assistant\n"""
    ids = tokenizer.encode(query2)
    unshare_token_len = len(ids)
    ids = ids + (UNSHARE_LENGTH - unshare_token_len) * [0]
    input_ids = torch.tensor(ids).view(UNSHARE_LENGTH).to(device)
    out = embed(input_ids).view(1, UNSHARE_LENGTH, HIDDEN_SIZE)
    position_ids = list(range(token_len, token_len + unshare_token_len)) + (UNSHARE_LENGTH - unshare_token_len) * [0]
    position_ids = torch.tensor([position_ids]).to(device)
    attention_mask = torch.ones((UNSHARE_LENGTH, UNSHARE_LENGTH + SHARE_LENGTH)).float() * -10000.0
    for i in range(unshare_token_len):
        for j in range(token_len):
            attention_mask[i][j] = 0.0
        for j in range(SHARE_LENGTH, SHARE_LENGTH + UNSHARE_LENGTH):
            if j - SHARE_LENGTH <= i:
                attention_mask[i][j] = 0.0
    attention_mask = attention_mask.view(1, 1, SHARE_LENGTH, UNSHARE_LENGTH + SHARE_LENGTH).to(device)
    for i in range(NUM_LAYERS):
        out, k, v = blocks_unshare[i](out, position_ids, attention_mask, k_cache[i][:,:SHARE_LENGTH], v_cache[i][:,:SHARE_LENGTH])
        k_cache[i][:,SHARE_LENGTH:SHARE_LENGTH + unshare_token_len] = k[:,:unshare_token_len]
        v_cache[i][:,SHARE_LENGTH:SHARE_LENGTH + unshare_token_len] = v[:,:unshare_token_len]
    out = out[:, unshare_token_len - 1:unshare_token_len].view(1, 1, HIDDEN_SIZE)
    lm = LmHead()
    greedy_head = GreedyHead()
    token = greedy_head(lm(out)).view(1)
    out_ids = [int(token)]
    word = tokenizer.decode([int(token)])
    print(word, end="")
    
    token_len = token_len + unshare_token_len

    # decode
    while int(token) != tokenizer.eos_token_id and token_len < SEQ_LENGTH:
        token_len += 1
        unshare_token_len += 1
        input_ids = torch.tensor([token]).to(device)
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]]).to(device)
        attention_mask = -10000 * torch.ones((1, 1, 1, SEQ_LENGTH + 1)).float().to(device)
        for i in range(ori_token_len):
            attention_mask[0,0,0,i] = 0
        for i in range(SHARE_LENGTH, SHARE_LENGTH + unshare_token_len):
            attention_mask[0,0,0,i] = 0
        attention_mask[:, :, :, SEQ_LENGTH] = 0
        # attention_mask = torch.zeros((1, 1, 1, SEQ_LENGTH + 1)).float().to(device)
        # attention_mask[:, :, :, token_len:SEQ_LENGTH] = -10000.0
        for i in range(NUM_LAYERS):
            out, k, v = block_kvs[i](out, position_ids, attention_mask, k_cache[i], v_cache[i])
            k_cache[i][:,unshare_token_len + SHARE_LENGTH:unshare_token_len + SHARE_LENGTH+1] = k
            v_cache[i][:,unshare_token_len + SHARE_LENGTH:unshare_token_len + SHARE_LENGTH+1] = v
        print(out)
        token = greedy_head(lm(out)).view(1)
        out_ids.append(int(token))
        word = tokenizer.decode([int(token)])
        print(word, end="")
        # np.save(f'torch_{token_len}.npy', out)
    print("\noutput_ids:{}".format(out_ids))
    

# test_net_with_mask()

# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# export models
print(f'Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
    convert_block(i)
    convert_block_unshare(i)
    convert_block_cache(i)

print(f'Convert embedding')
convert_embedding()

print(f'Convert lm_head')
convert_lm_head()
convert_greedy_head()
convert_penalty_sample_head()
