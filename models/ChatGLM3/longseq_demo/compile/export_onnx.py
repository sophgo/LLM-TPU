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
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
parser.add_argument('-s', '--seq_length', type=int, default=4096, help="sequence length")
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
args = parser.parse_args()

model_path = args.model_path
folder = f"./tmp/onnx"
device = torch.device(args.device)

origin_model = AutoModel.from_pretrained(
    model_path, trust_remote_code=True, attn_implementation='eager',torch_dtype=torch.float, device_map='auto').eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

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
if transformer.seq_length is not None:
    assert transformer.seq_length == args.seq_length
if config.seq_length is not None:
    assert config.seq_length == args.seq_length

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
        return hidden_states, present_k[1:], present_v[1:]


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.encoder.final_layernorm(hidden_states)
        m_logits = transformer.output_layer(hidden_states)
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

        # top_pc
        import pdb; pdb.set_trace()
        cumulative_probs = logits.softmax(dim=1).cumsum(dim=1)
        mask = cumulative_probs < top_p
        mask = mask + self.keep_matrix
        filtered_logits = torch.where(mask, logits, torch.FloatTensor([-1000.]))
        probs = filtered_logits.softmax(dim=1)
        return probs, token


def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((SEQ_LENGTH, 1, HIDDEN_SIZE), dtype = torch.float).to(device)
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = -1000 * torch.ones((1, 1, SEQ_LENGTH, SEQ_LENGTH), dtype = torch.float).triu(diagonal=1).to(device)
    # import  pdb; pdb.set_trace()
    # with torch.no_grad():
    #     traced_model = torch.jit.trace(model,(hidden_states, position_ids, attention_mask) )
    # traced_model.eval()
    # model.eval()
    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/block_{layer_id}_16k_dynamic.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        dynamic_axes={
        'input_states' : {0 : 'sequence_len'},  # 使得第0、2、3维度是动态的
        'position_ids' : {1 : 'sequence_len'},  # 使得第0、2、3维度是动态的
        'attention_mask' : {2 : 'sequence_len_row',3 : 'sequence_len_col'},  # 使得第0、2、3维度是动态的
        'hidden_states' : {0 : 'sequence_len'},
        'past_k' : {0 : 'sequence_len'},
        'past_v' : {0 : 'sequence_len'}
        },
        opset_version=15)
    
    # import onnx
    # from onnx2pytorch import ConvertModel

    # # 加载ONNX模型
    # onnx_model = onnx.load(f'{folder}/block_{layer_id}_dynamic.onnx')

    # # 转换ONNX模型为PyTorch模型
    # pytorch_model = ConvertModel(onnx_model)

    # # 保存PyTorch模型
    # torch.save(pytorch_model, f'{folder}/model_{layer_id}.pt')
    # model = Block(layer_id)
    # module = torch.jit.trace(model.forward, (hidden_states, position_ids, attention_mask))
    # torch.jit.save(module, f'{folder}/block0.pt')
    # model = Block(layer_id)
    # scripted_module = torch.jit.script(model)
    # torch.jit.save(scripted_module, f'{folder}/block_{layer_id}_dynamic.pt')

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
    embed = Embedding()
    blocks = [Block(i) for i in range(NUM_LAYERS)]
    block_kvs = [BlockCache(i) for i in range(NUM_LAYERS)]
    # ids = tokenizer.encode('你好')
    # system_prompt = "You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."
    # history = [{"role": "system", "content": system_prompt}]
    history = []
    visited_token = []
    query = '1 + 1'
    print(query)
    ids = tokenizer.build_chat_input(query, history=history, role="user")
    ids = ids["input_ids"][0].tolist()
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

    for i in range(NUM_LAYERS):
        # import numpy as np
        # np.savez("block0_input.npz", input_states=out, position_ids=position_ids, attention_mask=attention_mask)
        # if i == 20:
        #     import pdb; pdb.set_trace()
        out, kv_cache = blocks[i](out, position_ids, attention_mask)
        # import pdb; pdb.set_trace()
        k, v = kv_cache
        k[SEQ_LENGTH - token_len:] = k[:token_len]
        v[SEQ_LENGTH - token_len:] = v[:token_len]
        k[:SEQ_LENGTH - token_len] = 0
        v[:SEQ_LENGTH - token_len] = 0
        k_cache.append(k)
        v_cache.append(v)
    # import pdb; pdb.set_trace()
    out = out[token_len - 1:token_len].view(1, 4096)
    lm = LmHead()
    greedyhead = GreedyHead()
    lm_out = lm(out)
    token = greedyhead(lm_out)
    visited_token.append(int(token))
    out_ids = [int(token)]
    word = tokenizer._convert_id_to_token(int(token[0]))
    print(word, end="")
    import pdb; pdb.set_trace()
    while token > 2 and token_len < 64:
        token_len += 1
        # import pdb;pdb.set_trace()
        input_ids = torch.tensor([token])
        out = embed(input_ids).view(1, 1, 4096)
        position_ids = torch.tensor([[token_len - 1]])
        attention_mask = torch.ones((1, 1, 1, SEQ_LENGTH + 1))*-1000
        attention_mask[:, :, :, SEQ_LENGTH + 1 - token_len:] = 0
        for i in range(NUM_LAYERS):
            # if i == 27:
                # import pdb;pdb.set_trace()
            out, k_cache[i], v_cache[i] = block_kvs[i](out, position_ids,
                                                       attention_mask,
                                                       k_cache[i], v_cache[i])
            k_cache[i][:SEQ_LENGTH - token_len] = 0
            v_cache[i][:SEQ_LENGTH - token_len] = 0
        # import pdb;pdb.set_trace()
        lm_out = lm(out)
        token = greedyhead(lm_out)
        visited_token.append(int(token))
        out_ids.append(int(token))
        word = tokenizer._convert_id_to_token(int(token[0]))
        # import pdb; pdb.set_trace()
        print(word, end="")
    print("\noutput_ids:{}".format(out_ids))

# test_net_with_mask()
# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# export models
print(f'Convert block & block_cache')
# import pdb; pdb.set_trace()
for i in tqdm(range(NUM_LAYERS)):
    convert_block(i)
    convert_block_cache(i)
    # break

print(f'Convert embedding')
convert_embedding()

print(f'Convert lm_head')
convert_lm_head()
convert_greedy_head()
convert_penalty_sample_head()

