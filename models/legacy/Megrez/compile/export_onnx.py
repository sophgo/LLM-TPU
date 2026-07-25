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
from transformers import AutoTokenizer, AutoModelForCausalLM
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

torch.set_grad_enabled(False)


parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('-m', '--model_path', type=str, default="/workspace/Megrez-3B-Instruct", help='path to the torch model')
parser.add_argument('-s', '--seq_length', type=int, default=512, help="sequence length")
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
parser.add_argument('-n', '--num_threads', type=int, default=1, help='The number of threads used for torch if device is cpu')

args = parser.parse_args()

model_path = args.model_path
folder = f"./tmp/onnx"

# set
device = torch.device(args.device)
if args.device == "cpu":
    dtype = torch.float
    torch.set_num_threads(args.num_threads)
else:
    dtype = torch.bfloat16

origin_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, attn_implementation='eager', 
                                                    torch_dtype=dtype, device_map=args.device).eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

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
ID_EOS = [120005, 120000, 120025]

print(f'Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n')
# print(origin_model)
# breakpoint()


class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        return transformer.embed_tokens(input_ids)


class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        value_states = torch.randn(
            (1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)
        self.rotary_emb = self.layer.self_attn.rotary_emb
        self.cos, self.sin = self.rotary_emb(value_states, seq_len=SEQ_LENGTH)
        self.cos = self.cos.view(SEQ_LENGTH, HEAD_DIM)
        self.sin = self.sin.view(SEQ_LENGTH, HEAD_DIM)

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids,
                                            use_cache=True,
                                            position_embeddings=(self.cos, self.sin),
                                            )
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        value_states = torch.randn(
            (1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)
        self.rotary_emb = self.layer.self_attn.rotary_emb
        self.cos, self.sin = self.rotary_emb(value_states, seq_len=SEQ_LENGTH)
        self.cos = self.cos.view(SEQ_LENGTH, HEAD_DIM)
        self.sin = self.sin.view(SEQ_LENGTH, HEAD_DIM)

    def forward(self, hidden_states, position_ids, attention_mask, past_k, past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids=position_ids,
                                            past_key_value=(past_k, past_v),
                                            use_cache=True,
                                            position_embeddings=(self.cos, self.sin),
                                            )
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


def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.ones(
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
    attention_mask = torch.ones((1, 1, 1, SEQ_LENGTH + 1)).to(dtype).to(device)
    past_k = torch.randn(
        (1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)
    past_v = torch.randn(
        (1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)

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
    hidden_states = torch.randn(1, HIDDEN_SIZE).to(dtype).to(device)

    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head.pt')


# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

def test_net_with_mask():
    embed = Embedding()
    blocks = [Block(i) for i in range(NUM_LAYERS)]
    block_kvs = [BlockCache(i) for i in range(NUM_LAYERS)]
    lm = LmHead()

    msgs = [{'role': 'user', 'content': '你是什么架构的模型。'}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
    print("input ids:{}".format(ids))

    token_len = len(ids)
    ids = ids + (SEQ_LENGTH - token_len) * [0]
    input_ids = torch.tensor(ids).view(SEQ_LENGTH).to(device)
    out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)
    print(f"out: {out.shape}")

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
        out, k, v = blocks[i](out.to(dtype), position_ids,
                              attention_mask.to(dtype))
        k[:, :, token_len:, :] = 0
        v[:, :, token_len:, :] = 0
        k_cache.append(k)
        v_cache.append(v)

    out = out[:, token_len - 1:token_len].view(1, 1, HIDDEN_SIZE)
    
    token = lm(out.to(dtype)).view(1)
    out_ids = [int(token)]
    while int(token) not in ID_EOS and token_len < SEQ_LENGTH:
        token_len += 1
        input_ids = torch.tensor([token]).to(device)
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]]).to(device)
        attention_mask = torch.zeros((1, 1, 1, SEQ_LENGTH + 1)).float().to(device)
        attention_mask[:, :, :, token_len-1:SEQ_LENGTH] = -10000.0
        for i in range(NUM_LAYERS):
            out, k, v = block_kvs[i](out.to(dtype), position_ids, attention_mask.to(dtype),
                                     k_cache[i].to(dtype), v_cache[i].to(dtype))
            k_cache[i][:, token_len-1:token_len, :, :] = k[:, :, :, :]
            v_cache[i][:, token_len-1:token_len, :, :] = v[:, :, :, :]
        token = lm(out.to(dtype)).view(1)
        out_ids.append(int(token))
    words = tokenizer.decode(out_ids)
    print(words)
    print("\noutput_ids:{}".format(out_ids))


# test_net_with_mask()
# exit()

# export models
print('Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
    convert_block_cache(i)
    convert_block(i)

print('Convert embedding')
convert_embedding()

print('Convert lm_head')
convert_lm_head()
print("Done!")