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
import requests
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
parser.add_argument('-s', '--seq_length', type=int, default=512, help="sequence length")
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
parser.add_argument('-n', '--num_threads', type=int, default=1, help='The number of threads used for torch if device is cpu')
parser.add_argument('--lmhead_with_topk', type=int, default=0, help="only trace the LmHeadWithTopK")

args = parser.parse_args()

model_path = args.model_path
folder = "./tmp/onnx" 

device = torch.device(args.device)
if args.device == "cpu":
    dtype = torch.float
    torch.set_num_threads(args.num_threads)
else:
    dtype = torch.bfloat16

origin_model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True,
    torch_dtype=torch.float, device_map=args.device).eval()

for param in origin_model.parameters():
    param.requires_grad = False

config = origin_model.config
transformer = origin_model.llm.model
layers = transformer.layers

SEQ_LENGTH = args.seq_length
NUM_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
NUM_KEY_VALUE_HEADS = config.num_key_value_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
VOCAB_SIZE = config.vocab_size
IMAGE_SIZE = config.vision_config.image_size

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

class VisionTransformer(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, image, patch_attn_mask, tgt_sizes):
        out = origin_model.vpm(image, patch_attn_mask, tgt_sizes, return_dict=False)
        res = origin_model.resampler(out[0], tgt_sizes)
        return res

class LmHeadWithTopK(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.ln_f(hidden_states)
        m_logits = origin_model.llm.lm_head(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token

class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states)
        m_logits = origin_model.llm.lm_head(hidden_states)
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
        (1, SEQ_LENGTH, HIDDEN_SIZE)).float().to(device)
    position_ids = torch.tensor(
        [range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.randn(
        (1, 1, SEQ_LENGTH, SEQ_LENGTH)).float().to(device)
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
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).float().to(device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device)
    attention_mask = torch.ones(
        (1, 1, 1, SEQ_LENGTH + 1)).float().to(device)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).float().to(device)
    past_v = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).float().to(device)

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

def convert_vision_transformer():
    model = VisionTransformer()
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(dtype=torch.float32, device=device)

    torch.onnx.export(
        model, x,
        f'{folder}/vision_transformer.onnx',
        verbose=False,
        input_names=['input_images'],
        output_names=['output_images'],
        do_constant_folding=True,
        opset_version=15
    )

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
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).float().to(device)
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

# create folders to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# # export models
# print('Convert Vision Transformer')
# convert_vision_transformer()

# print('Convert block & block_cache')
# for i in tqdm(range(NUM_LAYERS)):
#     convert_block(i)
#     convert_block_cache(i)

# print('Convert embedding')
# convert_embedding()

# print('Convert lm_head')
# if args.lmhead_with_topk != 0:
#     convert_lm_head_with_topk()
# else:
#     convert_lm_head()
#     convert_greedy_head()
#     convert_penalty_sample_head()

# print("Done")

def test_torch():
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # image = Image.open('xx.jpg').convert('RGB')
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    question = 'What is in the image?'
    msgs = [{'role': 'user', 'content': [image, question]}]

    res = origin_model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        num_beams=1,
        do_sample=False
    )
    print(res)

def test_net_with_mask():
    def get_vllm_embedding(data):
        tgt_sizes = data['tgt_sizes']
        pixel_values_list = data['pixel_values']
        vision_hidden_states = []
        all_pixel_values = []
        img_cnt = []
        for pixel_values in pixel_values_list:
            img_cnt.append(len(pixel_values))
            all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])
        breakpoint()
        # exist image
        if all_pixel_values:
            tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
            tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

            max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

            all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,
                                                                padding_value=0.0)
            B, L, _ = all_pixel_values.shape
            all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

            patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
            for i in range(B):
                patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

            vision_batch_size = config.vision_batch_size
            all_pixel_values = all_pixel_values.type(dtype)
            if B > vision_batch_size:
                raise ValueError("not support now")
                # vision_embedding = vpm(all_pixel_values, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes).last_hidden_state
            # vision_embedding = resampler(vision_embedding, tgt_sizes)
            vision_embedding = vit(all_pixel_values, patch_attn_mask, tgt_sizes)

        vllm_embedding = embed(data['input_ids'])

        bs = len(data['input_ids'])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data['image_bound'][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)

                    cur_vllm_emb.scatter_(0, image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                                          cur_vs_hs.view(-1, cur_vs_hs.shape[-1]))
                elif self.training:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0

    # 1. Set model & tokenizer & processor
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    embed = Embedding()
    blocks = [Block(i) for i in range(NUM_LAYERS)]
    block_kvs = [BlockCache(i) for i in range(NUM_LAYERS)]
    vit = VisionTransformer()

    # 2. Set input
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    query = "(<image>./</image>)\nWhat is in the image?"
    msgs = [{"role": "user", "content": query}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    # SEQ_LENGTH = len(ids) + 20
    # print("input ids:{}".format(ids))
    # token_len = len(ids)
    # ids = ids + (SEQ_LENGTH - token_len) * [0]
    # input_ids = torch.tensor(ids).view(SEQ_LENGTH)

    # out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)
    # # breakpoint()
    # position_ids = list(range(token_len)) + (SEQ_LENGTH - token_len) * [0]
    # position_ids = torch.tensor([position_ids])
    # attention_mask = torch.ones((SEQ_LENGTH, SEQ_LENGTH)) * -10000.0
    # for i in range(token_len):
    #     attention_mask[i,:i+1] = 0
    # attention_mask = attention_mask.view((1,1,SEQ_LENGTH,SEQ_LENGTH))
    # k_cache = []
    # v_cache = []

    prompts_lists, input_images_lists = [], []
    prompts_lists.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    input_images_lists.append([image])
    inputs = processor(
        prompts_lists,
        input_images_lists,
        max_slice_nums=None,
        use_image_id=None,
        max_length=args.seq_length
    ).to(device)

    get_vllm_embedding(inputs)
    breakpoint()

    # 3. inference visition transform


    for i in tqdm(range(NUM_LAYERS)):
        out, k, v = blocks[i](out, position_ids, attention_mask)
        k[:,token_len:,:,:] = 0
        v[:,token_len:,:,:] = 0
        k_cache.append(k)
        v_cache.append(v)
    out = out[0, token_len - 1:token_len, :].view(1, HIDDEN_SIZE)
    lm = LmHead()
    m_logits = lm(out)
    greedy_head = GreedyHead()
    token = greedy_head(m_logits)
    out_ids = [int(token)]
    word = tokenizer._convert_id_to_token(int(token[0]))
    print(word, end="")
    for i in tqdm(range(10)):
        token_len += 1
        input_ids = torch.tensor([token])
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]])
        attention_mask = torch.ones((1, 1, 1, SEQ_LENGTH + 1)) * -10000.0
        attention_mask[:, :, :, :token_len] = 0
        attention_mask[:, :, :, -1] = 0

        for i in range(NUM_LAYERS):
            out, k_cache_present, v_cache_present = block_kvs[i](out, position_ids,
                                                       attention_mask,
                                                       k_cache[i], v_cache[i])
            k_cache[i][:,token_len-1:token_len,:,:] = k_cache_present
            v_cache[i][:,token_len-1:token_len,:,:] = v_cache_present
        m_logits = lm(out)
        token = greedy_head(m_logits)
        out_ids.append(int(token))
        word = tokenizer._convert_id_to_token(int(token[0]))
        print(word, end="")
    print("\noutput_ids:{}".format(out_ids))


test_torch()
# test_net_with_mask()