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
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
parser.add_argument('-s', '--seq_length', type=int, default=512, help="sequence length")
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
parser.add_argument('--lmhead_with_topk', type=int, default=0, help="only trace the LmHeadWithTopK")

args = parser.parse_args()

model_path = args.model_path
folder = "./tmp/onnx"

device = torch.device(args.device)
if args.device == "cpu":
    dtype = torch.float
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    dtype = torch.bfloat16

origin_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="auto",
    attn_implementation="eager").eval()

for param in origin_model.parameters():
    param.requires_grad = False

config = origin_model.config
transformer = origin_model.model
layers = transformer.layers

SEQ_LENGTH = args.seq_length
NUM_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
NUM_KEY_VALUE_HEADS = config.num_key_value_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
VOCAB_SIZE = config.vocab_size

print(f'\
Layers: {NUM_LAYERS}\n\
Hidden size: {HIDDEN_SIZE}\n\
Head dim: {HEAD_DIM}\n\
Q Heads: {NUM_ATTENTION_HEADS}\n\
KV Heads: {NUM_KEY_VALUE_HEADS}\n\
Seq length: {SEQ_LENGTH}\n')

class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        out = transformer.embed_tokens(input_ids)
        return out.float()


class Block(torch.nn.Module):

    def __init__(self,):
        super().__init__()
        self.layers = layers
        self.rotary_emb = self.layers[0].self_attn.rotary_emb
        value_states = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM))
        position_ids = torch.tensor([range(SEQ_LENGTH)],dtype=torch.long)
        self.cos, self.sin = self.rotary_emb(value_states, position_ids)

    def forward(self,
                hidden_states,
                position_ids,
                attention_mask):
        past_ks = []
        past_vs = []
        for i in range(NUM_LAYERS):
            hidden_states, past_kv = self.layers[i](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                position_embeddings=(self.cos, self.sin))
            past_ks.append(past_kv[0])
            past_vs.append(past_kv[1])
        return hidden_states, past_ks, past_vs


class BlockCache(torch.nn.Module):

    def __init__(self,):
        super().__init__()
        self.layers = layers
        self.rotary_emb = self.layers[0].self_attn.rotary_emb
        value_states = torch.randn((1, 1, NUM_KEY_VALUE_HEADS, HEAD_DIM))
        position_ids = torch.tensor([range(SEQ_LENGTH)],dtype=torch.long)
        self.cos, self.sin = self.rotary_emb(value_states, position_ids)

    def forward(self,
                hidden_states,
                position_ids,
                attention_mask,
                history_ks,
                history_vs):
        past_ks = []
        past_vs = []
        for i in range(NUM_LAYERS):
            hidden_states, past_kv = self.layers[i](
                hidden_states,
                past_key_value=(history_ks[i], history_vs[i]),
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True,
                position_embeddings=(self.cos, self.sin))
            past_ks.append(past_kv[0])
            past_vs.append(past_kv[1])
        return hidden_states, past_ks, past_vs


class LmHeadWithTopK(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states)
        m_logits = origin_model.lm_head(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token


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


def convert_block():
    model = Block()
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE)).to(dtype)
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long)
    attention_mask = torch.randn((1, 1, SEQ_LENGTH, SEQ_LENGTH)).to(dtype)

    input_names = ['input_states', 'position_ids', 'attention_mask']
    output_names = ['hidden_states']
    for i in range(NUM_LAYERS):
        output_names.append(f'past_k{i}')
    for i in range(NUM_LAYERS):
        output_names.append(f'past_v{i}')

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/block/block.onnx',
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        opset_version=15)


def convert_block_cache():
    model = BlockCache()
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).to(dtype)
    position_ids = torch.tensor([range(1)], dtype=torch.long)
    attention_mask = torch.ones((1, 1, 1, SEQ_LENGTH + 1)).to(dtype)
    history_ks = []
    history_vs = []
    for i in range(NUM_LAYERS):
        history_ks.append(torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)))
        history_vs.append(torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)))
    input_names = ['input_states', 'position_ids', 'attention_mask']
    output_names = ['hidden_states']
    for i in range(NUM_LAYERS):
        input_names.append(f'history_k{i}')
        output_names.append(f'past_k{i}')
    for i in range(NUM_LAYERS):
        input_names.append(f'history_v{i}')
        output_names.append(f'past_v{i}')

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask, history_ks, history_vs),
        f'{folder}/cache/cache.onnx',
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
        opset_version=15)


def convert_embedding():
    model = Embedding()
    input_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.int32).to(device)
    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f'{folder}/embedding.pt')


def convert_lm_head_with_topk():
    model = LmHeadWithTopK()
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).float().to(device)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head_with_topk.pt')


def convert_lm_head():
    model = LmHead()
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).to(dtype).to(device)
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
    return f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'


def test_net_with_mask():
    embed = Embedding().to(device)
    fullblock = Block()
    fullcache = BlockCache()
    lm = LmHead()
    greedy_head = GreedyHead()
    query = """tell me about sophgo in ten word"""
    print(query)
    promt = build_prompt(query)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ids = tokenizer.encode(promt)
    print("input ids:{}".format(ids))

    token_len = len(ids)
    ori_token_len = token_len
    ids = ids + (SEQ_LENGTH - token_len) * [0]
    input_ids = torch.tensor(ids).view(SEQ_LENGTH).to(device)
    out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)
    position_ids = list(range(token_len)) + (SEQ_LENGTH - token_len) * [0]
    position_ids = torch.tensor([position_ids]).to(device)
    attention_mask = torch.ones((SEQ_LENGTH, SEQ_LENGTH)).float() * -10000.0
    for i in range(token_len):
        for j in range(token_len):
            if j <= i:
                attention_mask[i][j] = 0.0
    attention_mask = attention_mask.view(1, 1, SEQ_LENGTH, SEQ_LENGTH).to(device)
    out, k_caches, v_caches = fullblock(out, position_ids, attention_mask)
    out = out[:, token_len - 1:token_len].view(1, 1, HIDDEN_SIZE)
    token = greedy_head(lm(out)).view(1)
    out_ids = []
    while int(token) != tokenizer.eos_token_id and token_len <= ori_token_len + 10:
        word = tokenizer.decode([int(token)])
        print(word, end="")
        token_len += 1
        out_ids.append(int(token))
        input_ids = torch.tensor([token]).to(device)
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]]).to(device)
        attention_mask = torch.zeros((1, 1, 1, SEQ_LENGTH + 1)).float().to(device)
        attention_mask[:, :, :, token_len:SEQ_LENGTH] = -10000.0
        out, k, v = fullcache(out,
                              position_ids,
                              attention_mask,
                              k_caches,
                              v_caches)
        for i in range(NUM_LAYERS):
            k_caches[i][:, token_len-1:token_len] = k[i]
            v_caches[i][:, token_len-1:token_len] = v[i]
        token = greedy_head(lm(out.to(dtype))).view(1)
    print("\noutput_ids:{}".format(out_ids))


# test_net_with_mask()

if not os.path.exists(folder):
    os.makedirs(folder)
    os.makedirs(folder+'/block')
    os.makedirs(folder+'/cache')

print('Convert block & block_cache')
convert_block()
convert_block_cache()

print('Convert embedding')
convert_embedding()

print('Convert lm_head')
if args.lmhead_with_topk:
    convert_lm_head_with_topk()
else:
    convert_lm_head()
    convert_greedy_head()
    convert_penalty_sample_head()
