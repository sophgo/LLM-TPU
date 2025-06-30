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
    attn_implementation='eager',
    torch_dtype=dtype,
    device_map="auto").eval()

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
QK_HEAD_DIM = config.qk_nope_head_dim + config.qk_rope_head_dim
V_HEAD_DIM = config.hidden_size // config.num_attention_heads
VOCAB_SIZE = config.vocab_size

print(f'\
Layers: {NUM_LAYERS}\n\
Hidden size: {HIDDEN_SIZE}\n\
Q Heads: {NUM_ATTENTION_HEADS}\n\
KV Heads: {NUM_KEY_VALUE_HEADS}\n\
QK Head dim: {QK_HEAD_DIM}\n\
V Head dim: {V_HEAD_DIM}\n\
Seq length: {SEQ_LENGTH}\n')

class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        out = transformer.embed_tokens(input_ids) * config.scale_emb
        return out.float()


class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = self.layer.self_attn.rotary_emb
        value_states = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, V_HEAD_DIM)).to(dtype).to(device)
        self.cos, self.sin = self.rotary_emb(value_states, seq_len=SEQ_LENGTH)

    def forward(self,
                hidden_states,
                position_ids,
                attention_mask):
        hidden_states, past_kv = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            position_embeddings=(self.cos, self.sin))
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = self.layer.self_attn.rotary_emb
        value_states = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, V_HEAD_DIM)).to(dtype).to(device)
        self.cos, self.sin = self.rotary_emb(value_states, seq_len=SEQ_LENGTH)

    def forward(self,
                hidden_states,
                position_ids,
                attention_mask,
                past_k, past_v):
        hidden_states, past_kv = self.layer(
            hidden_states,
            past_key_value=(past_k, past_v),
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            position_embeddings=(self.cos, self.sin))
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class LmHeadWithTopK(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = config.dim_model_base / config.hidden_size

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states) * self.scale
        m_logits = origin_model.lm_head(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = config.dim_model_base / config.hidden_size

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states) * self.scale
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


def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn(
        (1, SEQ_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor(
        [range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.randn(
        (1, 1, SEQ_LENGTH, SEQ_LENGTH)).to(dtype).to(device)

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
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device)
    attention_mask = torch.ones(
        (1, 1, 1, SEQ_LENGTH + 1)).to(dtype).to(device)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, QK_HEAD_DIM)).to(dtype).to(device)
    past_v = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, V_HEAD_DIM)).to(dtype).to(device)

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


def test_net_with_mask():
    embed = Embedding().to(device)
    blocks = [Block(i).to(device) for i in range(NUM_LAYERS)]
    block_kvs = [BlockCache(i).to(device) for i in range(NUM_LAYERS)]
    lm = LmHead()
    greedy_head = GreedyHead()
    query = "hello"
    messages = [
        {"role": "user", "content": query},
    ]
    print(query)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
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
    k_cache = []
    v_cache = []
    for i in range(NUM_LAYERS):
        out[:,token_len:] = 0
        if i == 0:
            import numpy
            data={'input_0':out.numpy(),'input_1':position_ids.numpy().astype(numpy.int32),'input_2':attention_mask.numpy()}
            numpy.savez("in1.npz",**data)
        out, k, v = blocks[i](out.to(dtype),
                              position_ids,
                              attention_mask)
        if i == 0:
            import numpy
            data={'output_0':out.numpy(),'output_1':k.numpy(),'output_2':v.numpy()}
            numpy.savez("ref.npz",**data)
        k_cache.append(k)
        v_cache.append(v)
    out = out[:, token_len - 1:token_len].view(1, 1, HIDDEN_SIZE)
    token = int(greedy_head(lm(out.to(dtype))).view(1))
    out_ids = [token]
    while token != tokenizer.eos_token_id:
        pre_word = tokenizer.decode([token], skip_special_tokens=True)
        word = tokenizer.decode([token, token], skip_special_tokens=True)[len(pre_word):]
        print(word, flush=True, end="")
        token_len += 1
        input_ids = torch.tensor([token]).to(device)
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]]).to(device)
        attention_mask = torch.zeros((1, 1, 1, SEQ_LENGTH + 1)).float().to(device)
        attention_mask[:, :, :, token_len:SEQ_LENGTH] = -10000.0
        for i in range(NUM_LAYERS):
            out, k, v = block_kvs[i](out.to(dtype),
                                     position_ids,
                                     attention_mask,
                                     k_cache[i].to(dtype),
                                     v_cache[i].to(dtype))
            k_cache[i][:, token_len:token_len+1] = k
            v_cache[i][:, token_len:token_len+1] = v
        token = int(greedy_head(lm(out.to(dtype))).view(1))
        out_ids.append(token)
    print("\noutput_ids:{}".format(out_ids))


# test_net_with_mask()

# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# export models
print('Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
   convert_block(i)
   convert_block_cache(i)

print('Convert embedding')
convert_embedding()

print('Convert lm_head')
if args.lmhead_with_topk:
    convert_lm_head_with_topk()
else:
    convert_lm_head()
    convert_greedy_head()
    convert_penalty_sample_head()
