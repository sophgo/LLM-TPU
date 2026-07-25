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
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
parser.add_argument('-s', '--seq_length', type=int, default=512, help="sequence length")
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
parser.add_argument('-n', '--num_threads', type=int, default=1, help='The number of threads used for torch if device is cpu')
parser.add_argument('--lmhead_with_topk', type=int, default=0, help="only trace the LmHeadWithTopK")

args = parser.parse_args()

model_path = args.model_path
folder = f"./tmp/onnx"

device = torch.device(args.device)
if device == 'cpu':
    torch.set_num_threads(args.num_threads)

origin_model = AutoModel.from_pretrained(
    model_path, trust_remote_code=True, attn_implementation='eager',torch_dtype=torch.float, device_map='auto').eval()

for param in origin_model.parameters():
    param.requires_grad = False

config = origin_model.config
transformer = origin_model.transformer
layers = transformer.encoder.layers

SEQ_LENGTH = transformer.seq_length
NUM_LAYERS = config.num_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
MULTI_QUERY_GROUP_NUM = config.multi_query_group_num
VOCAB_SIZE = config.vocab_size

print(f'Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n')
if transformer.seq_length is not None:
    assert transformer.seq_length == args.seq_length
if config.seq_length is not None:
    assert config.seq_length == args.seq_length

def setup_environment():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return

class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        return transformer.embedding.word_embeddings(input_ids)


class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_pos_emb = transformer.rotary_pos_emb(SEQ_LENGTH)

    def forward(self, hidden_states, position_ids, attention_mask):
        rotary_pos_emb = self.rotary_pos_emb[position_ids]
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            rotary_pos_emb=rotary_pos_emb)
        return hidden_states, past_kv


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_pos_emb = transformer.rotary_pos_emb(SEQ_LENGTH)

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        rotary_pos_emb = self.rotary_pos_emb[position_ids]
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            kv_cache=(past_k, past_v),
                                            rotary_pos_emb=rotary_pos_emb)
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.encoder.final_layernorm(hidden_states)
        m_logits = transformer.output_layer(hidden_states)
        return m_logits


class LmHeadWithTopK(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.encoder.final_layernorm(hidden_states)
        m_logits = transformer.output_layer(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token


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
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE), dtype = torch.float).to(device)
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = -1000 * torch.ones((1, 1, SEQ_LENGTH, SEQ_LENGTH), dtype = torch.float).triu(diagonal=1).to(device)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15,
	export_params=True)


def convert_block_cache(layer_id):
    model = BlockCache(layer_id)
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE), dtype = torch.float).to(device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device)
    attention_mask = torch.ones((1, 1, 1, SEQ_LENGTH + 1), dtype = torch.float).triu(diagonal=1).to(device)
    past_k = torch.randn((1, SEQ_LENGTH, MULTI_QUERY_GROUP_NUM, HEAD_DIM), dtype = torch.float).to(device)
    past_v = torch.randn((1, SEQ_LENGTH, MULTI_QUERY_GROUP_NUM, HEAD_DIM), dtype = torch.float).to(device)
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
    input_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.int32).to(device)
    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f'{folder}/embedding.pt')


def convert_lm_head():
    model = LmHead()
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).float().to(device)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head.pt')


def convert_lm_head_with_topk():
    model = LmHeadWithTopK()
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).float().to(device)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head_with_topk.pt')


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
    max_new_tokens = 10

    embed = Embedding()
    blocks = [Block(i) for i in range(NUM_LAYERS)]
    block_kvs = [BlockCache(i) for i in range(NUM_LAYERS)]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    visited_token = []
    query = 'hello'
    print(query)
    ids = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                        add_generation_prompt=True,
                                        tokenize=True,
                                        return_tensors="pt",
                                        return_dict=True
                                        )
    ids = ids["input_ids"][0].tolist()
    visited_token = visited_token + ids
    print("input ids:{}".format(ids))
    token_len = len(ids)
    ids = ids + (SEQ_LENGTH - token_len) * [0]
    input_ids = torch.tensor(ids).view(SEQ_LENGTH)
    out = embed(input_ids).view(1, SEQ_LENGTH, 4096)
    position_ids = list(range(token_len)) + (SEQ_LENGTH - token_len) * [0]
    position_ids = torch.tensor([position_ids])
    attention_mask = torch.full((SEQ_LENGTH, SEQ_LENGTH), -1000.0)
    for i in range(token_len):
        for j in range(token_len):
            if j <= i:
                attention_mask[i][j] = 0
    attention_mask = attention_mask.view(1, 1, SEQ_LENGTH, SEQ_LENGTH)
    k_cache = []
    v_cache = []

    for i in tqdm(range(NUM_LAYERS)):
        outputs = blocks[i](out, position_ids, attention_mask)
        # np.savez(f"block_{i}.npz", input_0=out, input_1=position_ids, input_2=attention_mask, output_0=outputs[0], output_1=outputs[1][0], output_2=outputs[1][1])
        out, kvcache = outputs
        k, v = kvcache
        k_cache.append(k)
        v_cache.append(v)
    out = out[0, token_len - 1:token_len].view(1, 4096)
    lm = LmHead()
    greedyhead = GreedyHead()
    lm_out = lm(out)
    token = greedyhead(lm_out)
    visited_token.append(int(token))
    out_ids = [int(token)]
    word = tokenizer._convert_id_to_token(int(token[0]))
    print(word, end="")
    while token_len < token_len + max_new_tokens:
        token_len += 1
        input_ids = torch.tensor([token])
        out = embed(input_ids).view(1, 1, 4096)
        position_ids = torch.tensor([[token_len - 1]])
        attention_mask = torch.ones((1, 1, 1, SEQ_LENGTH + 1))
        attention_mask[:, :, :, SEQ_LENGTH + 1 - token_len:] = 0
        for i in tqdm(range(NUM_LAYERS)):
            out, present_k, present_v = block_kvs[i](out, position_ids,
                                                       attention_mask,
                                                       k_cache[i], v_cache[i])
            k_cache[i][0, token_len:token_len+1] = present_k
            v_cache[i][0, token_len:token_len+1] = present_v
        lm_out = lm(out)
        token = greedyhead(lm_out)
        visited_token.append(int(token))
        out_ids.append(int(token))
        word = tokenizer._convert_id_to_token(int(token[0]))
        print(int(token), tokenizer.decode(token[0,0], skip_special_tokens=True))
    # print("\noutput_ids:{}".format(out_ids))
    print("\noutput_words:{}".format(tokenizer.decode(out_ids, skip_special_tokens=True)))

# test_net_with_mask()
setup_environment()

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
if args.lmhead_with_topk:
    convert_lm_head_with_topk()
else:
    convert_lm_head()
    convert_greedy_head()
    convert_penalty_sample_head()
