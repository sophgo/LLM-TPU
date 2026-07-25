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
import numpy
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='export onnx.')
parser.add_argument('-m', '--model_path', type=str, help='path to the torch model.')
parser.add_argument('-s', '--seq_length', type=int, default=1024, help="sequence length")
parser.add_argument('-i', '--image_size', type=int, default=384)
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
parser.add_argument('-f', '--folder', type=str, default='./tmp/onnx')
args = parser.parse_args()

device = torch.device(args.device)
if args.device == "cpu":
    dtype = torch.float
else:
    dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
	trust_remote_code=True,
    attn_implementation='eager',
    torch_dtype=dtype,
    device_map=device)
processor = AutoProcessor.from_pretrained(
    args.model_path,
    trust_remote_code=True)

# generate fake inputs for vit
image = Image.new('RGB', (args.image_size, args.image_size))
inputs = processor.process(images=[image], text="")
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

folder = args.folder
if not os.path.exists(folder):
    os.makedirs(folder)
    os.makedirs(folder+'/vit')

origin_model = model.eval()
for param in origin_model.parameters():
    param.requires_grad = False

config = origin_model.config
llm = origin_model.model
transformer = llm.transformer
vision = llm.vision_backbone
layers = transformer.blocks

# text config
SEQ_LENGTH = args.seq_length
NUM_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
NUM_KEY_VALUE_HEADS = config.num_key_value_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
VOCAB_SIZE = config.vocab_size
print(f'Layers: {NUM_LAYERS}\n\
Hidden size: {HIDDEN_SIZE}\n\
Query heads: {NUM_ATTENTION_HEADS}\n\
KV heads: {NUM_KEY_VALUE_HEADS}\n')

class VisionTransformer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.image_input_idx = inputs['image_input_idx'].reshape(1,-1)
        self.valid = self.image_input_idx >= 0
        self.image_idx = self.image_input_idx[self.valid]

    def forward(self, hidden_states, images, image_masks):
        image_features = vision(images, image_masks)[0]
        image_features = image_features.reshape(1, -1, HIDDEN_SIZE)
        hidden_states[0, self.image_idx] += image_features[self.valid]
        return hidden_states


class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        return transformer.wte(input_ids)


class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_bias=attention_mask,
                                            position_ids=position_ids,
                                            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask, past_k, past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_bias=attention_mask,
                                            position_ids=position_ids,
                                            layer_past=(past_k, past_v),
                                            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.ln_f(hidden_states) 
        m_logits = transformer.ff_out(hidden_states)
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


def convert_vision_transformer():
    model = VisionTransformer()
    input_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
    images = inputs['images'].to(dtype).to(device)
    image_masks = inputs['image_masks'].to(dtype).to(device)

    torch.onnx.export(
        model, (input_states, images, image_masks),
        f'{folder}/vit/vision_transformer.onnx',
        verbose=False,
        input_names=["input_states", "images", "image_masks"],
        output_names=['hidden_states'],
        do_constant_folding=True,
    )

def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.ones(1, 1, SEQ_LENGTH, SEQ_LENGTH).to(dtype).to(device)
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
    past_k = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM))
    past_v = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM))

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

def convert_greedy_head():   
    model = GreedyHead()
    m_logits = torch.randn(1, VOCAB_SIZE).to(dtype).to(device)

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
    m_logits = torch.randn(1, VOCAB_SIZE).to(dtype).to(device)
    input_ids = torch.tensor([range(SEQ_LENGTH)]).to(device)
    top_p = torch.tensor([0.8]).to(device)
    temperature = torch.tensor([0.98]).to(device)
    penalty = torch.tensor([0.98]).to(device)

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

def test_net_with_mask(image_path):
    image = Image.open(image_path).resize((args.image_size, args.image_size))
    query = 'What does this picture describe'
    print(f'image: "{image_path}"')
    print(f'query: "{query}"\n')
    inputs = processor.process(
        images=[image],
        text=query
    )
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # init inputs
    token_len = inputs['input_ids'].shape[1]
    ori_token_len = token_len
    input_ids = torch.zeros(SEQ_LENGTH).unsqueeze(0).to(torch.long)
    input_ids[:,:token_len] = inputs['input_ids'].to(torch.long)
    position_ids = list(range(token_len)) + (SEQ_LENGTH - token_len) * [0]
    position_ids = torch.tensor([position_ids])
    attention_mask = torch.ones((SEQ_LENGTH, SEQ_LENGTH)).float() * -10000.0
    for i in range(token_len):
        for j in range(token_len):
            if j <= i:
                attention_mask[i][j] = 0.0
    attention_mask = attention_mask.view(1, 1, SEQ_LENGTH, SEQ_LENGTH)

    # init models
    vit = VisionTransformer()
    embed = Embedding()
    lm_head = LmHead()
    greedy_head = GreedyHead()
    blocks = []
    block_kvs = []
    for i in range(NUM_LAYERS):
        blocks.append(Block(i))
        block_kvs.append(BlockCache(i))

    # inference
    out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)
    out = vit(out, inputs['images'], inputs['image_masks'])
    k_cache = []
    v_cache = []
    for i in range(NUM_LAYERS):
        out, k, v = blocks[i](out, position_ids, attention_mask)
        k_cache.append(k)
        v_cache.append(v)
    out = out[:, token_len - 1: token_len].view(1, 1, HIDDEN_SIZE)
    token = greedy_head(lm_head(out)).view(1)
    out_ids = []
    while int(token) != processor.tokenizer.eos_token_id:
        out_ids.append(int(token))
        word = processor.tokenizer.decode([int(token)])
        print(word, end="")
        token_len += 1
        input_ids = torch.tensor([token])
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]])
        attention_mask = torch.zeros((1, 1, 1, SEQ_LENGTH + 1)).float()
        attention_mask[:, :, :, token_len:SEQ_LENGTH] = -10000.0
        for i in range(NUM_LAYERS):
            out, k, v = block_kvs[i](out, position_ids, attention_mask, k_cache[i], v_cache[i])
            k_cache[i][:,token_len:token_len+1] = k
            v_cache[i][:,token_len:token_len+1] = v
        token = greedy_head(lm_head(out)).view(1)
    print("\noutput_ids:{}".format(out_ids))

# test_net_with_mask('../python_demo/test.jpg')

print(f'Convert vision transformer')
convert_vision_transformer()

print(f'Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
    convert_block(i)
    convert_block_cache(i)
 
print(f'Convert embedding')
convert_embedding()

print(f'Convert lm_head')
convert_lm_head()
convert_greedy_head()
convert_penalty_sample_head()
print("Done")
