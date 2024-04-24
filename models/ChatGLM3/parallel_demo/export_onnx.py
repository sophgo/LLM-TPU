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
import importlib.metadata
from transformers import AutoModel
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
parser.add_argument('-s', '--seq_length', type=int, default=512, help="sequence length")
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
args = parser.parse_args()

model_path = args.model_path
folder = f"./tmp/onnx"
device = torch.device(args.device)

origin_model = AutoModel.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.float, device_map='auto').eval()

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
VOCAB_SIZE = config.vocab_size

print(f'Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n')

transformers_version = importlib.metadata.version('transformers')
if transformers_version != config.transformers_version:
    raise ValueError(f"Your version of transformers is {transformers_version}, not {config.transformers_version}")

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

    def forward(self, hidden_states, position_ids, attention_mask):
        rotary_pos_emb = transformer.rotary_pos_emb(SEQ_LENGTH)[position_ids]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            rotary_pos_emb=rotary_pos_emb)
        return hidden_states, past_kv


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        rotary_pos_emb = transformer.rotary_pos_emb(SEQ_LENGTH)[position_ids]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
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
        _, token = torch.topk(m_logits.float(), 1)
        return token


def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((SEQ_LENGTH, 1, HIDDEN_SIZE), dtype = torch.float).to(device)
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = -1000 * torch.ones((1, 1, SEQ_LENGTH, SEQ_LENGTH), dtype = torch.float).triu(diagonal=1).to(device)

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
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE), dtype = torch.float).to(device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device)
    attention_mask = -1000 * torch.ones((1, 1, 1, SEQ_LENGTH + 1), dtype = torch.float).triu(diagonal=1).to(device)
    past_k = torch.randn((SEQ_LENGTH, 1, 2, HEAD_DIM), dtype = torch.float).to(device)
    past_v = torch.randn((SEQ_LENGTH, 1, 2, HEAD_DIM), dtype = torch.float).to(device)

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
    input = torch.randn(1, HIDDEN_SIZE).to(device)
    
    torch.onnx.export(model, (input),
                      f'{folder}/lm_head.onnx',
                      verbose=False,
                      input_names=['hidden_states'],
                      output_names=['token'],
                      do_constant_folding=True,
                      opset_version=15)

def test_net_with_mask():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    embed = Embedding()
    blocks = [Block(i) for i in range(NUM_LAYERS)]
    block_kvs = [BlockCache(i) for i in range(NUM_LAYERS)]
    # ids = tokenizer.encode('你好')
    # system_prompt = "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."
    # history = [{"role": "system", "content": system_prompt}]
    history = []
    visited_token = []
    query = 'can you help me'
    print(query)
    ids = tokenizer.build_chat_input(query, history=history, role="user")
    ids = ids["input_ids"][0].tolist()
    # import pdb; pdb.set_trace()
    visited_token = visited_token + ids
    print("input ids:{}".format(ids))
    token_len = len(ids)
    ids = ids + (SEQ_LENGTH - token_len) * [0]
    input_ids = torch.tensor(ids).view(SEQ_LENGTH)
    out = embed(input_ids).view(SEQ_LENGTH, 1, 4096)
    position_ids = list(range(token_len)) + (SEQ_LENGTH - token_len) * [0]
    position_ids = torch.tensor([position_ids])
    attention_mask = torch.full((SEQ_LENGTH, SEQ_LENGTH), -1000.0)
    for i in range(token_len):
        for j in range(token_len):
            if j <= i:
                attention_mask[i][j] = 0
    attention_mask = attention_mask.view(1, 1, SEQ_LENGTH, SEQ_LENGTH)
    # attention_mask = None
    k_cache = []
    v_cache = []

    for i in range(1):
        out, kv_cache = blocks[i](out, position_ids, attention_mask)
        k, v = kv_cache
        # k[SEQ_LENGTH - token_len:] = k[:token_len]
        # v[SEQ_LENGTH - token_len:] = v[:token_len]
        # k[:SEQ_LENGTH - token_len] = 0
        # v[:SEQ_LENGTH - token_len] = 0
        k_cache.append(k)
        v_cache.append(v)
    # import pdb; pdb.set_trace()
    import pdb;pdb.set_trace()
    out = out[token_len - 1:token_len].view(1, 4096)
    lm = LmHead()
    token = lm(out)
    visited_token.append(int(token))
    out_ids = [int(token)]
    word = tokenizer._convert_id_to_token(int(token[0]))
    print(word, end="")
    while token > 2 and token_len < 640:
        token_len += 1
        # import pdb;pdb.set_trace()
        input_ids = torch.tensor([token])
        out = embed(input_ids).view(1, 1, 4096)
        position_ids = torch.tensor([[token_len - 1]])
        attention_mask = torch.ones((1, 1, 1, SEQ_LENGTH + 1))*-1000
        attention_mask[:, :, :, :token_len-1] = 0
        attention_mask[:, :, :, -1] = 0
        for i in range(1):
            out, present_k_cache, present_v_cache = block_kvs[i](out, position_ids,
                                                       attention_mask,
                                                       k_cache[i], v_cache[i])
            import pdb;pdb.set_trace()
            k_cache[i][token_len-1] = present_k_cache
            v_cache[i][token_len-1] = present_v_cache
        # import pdb;pdb.set_trace()
        lm = LmHead()
        token = lm(out)
        visited_token.append(int(token))
        out_ids.append(int(token))
        word = tokenizer._convert_id_to_token(int(token[0]))
        import pdb; pdb.set_trace()
        print(word, end="")
    print("\noutput_ids:{}".format(out_ids))

test_net_with_mask()
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
