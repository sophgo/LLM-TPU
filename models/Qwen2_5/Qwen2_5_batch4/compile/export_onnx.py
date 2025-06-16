#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
# python3 export_onnx.py --model_path /data/pengchao.hu/workspace/Qwen2.5-3B-Instruct
import os
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import numpy as np

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='export onnx')
# yapf: disable
parser.add_argument('-m', '--model_path', required=True, type=str, help='path to the torch model')
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cuda")
parser.add_argument('--share_length', type=int, default=4096, help="share length")
parser.add_argument('--unshare_length', type=int, default=4096, help="unshare length")
# yapf: enable

args = parser.parse_args()

model_path = args.model_path
folder = "./tmp/onnx"

device = torch.device(args.device)
if args.device == "cpu":
    dtype = torch.float
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    dtype = torch.bfloat16

origin_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                    trust_remote_code=True,
                                                    attn_implementation='eager',
                                                    torch_dtype=dtype,
                                                    device_map="auto").eval()

for param in origin_model.parameters():
    param.requires_grad = False

config = origin_model.config
transformer = origin_model.model
layers = transformer.layers

SHARE_LENGTH = args.share_length
UNSHARE_LENGTH = args.unshare_length
SEQ_LENGTH = SHARE_LENGTH + UNSHARE_LENGTH
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

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = self.layer.self_attn.rotary_emb
        value_states = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS,
                                    HEAD_DIM)).to(dtype).to(device)
        position_ids = torch.tensor([range(SEQ_LENGTH)],
                                    dtype=torch.long).to(device)
        self.cos, self.sin = self.rotary_emb(value_states, position_ids)
        self.cos = self.cos.view(SEQ_LENGTH, HEAD_DIM)
        self.sin = self.sin.view(SEQ_LENGTH, HEAD_DIM)

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            use_cache=True,
                                            position_embeddings=(self.cos,
                                                                 self.sin))
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = self.layer.self_attn.rotary_emb
        value_states = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS,
                                    HEAD_DIM)).to(dtype).to(device)
        position_ids = torch.tensor([range(SEQ_LENGTH)],
                                    dtype=torch.long).to(device)
        self.cos, self.sin = self.rotary_emb(value_states, position_ids)
        self.cos = self.cos.view(SEQ_LENGTH, HEAD_DIM)
        self.sin = self.sin.view(SEQ_LENGTH, HEAD_DIM)

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            past_key_value=(past_k, past_v),
                                            position_ids=position_ids,
                                            attention_mask=attention_mask,
                                            use_cache=True,
                                            position_embeddings=(self.cos,
                                                                 self.sin))
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


def convert_block_share(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn(
        (1, SHARE_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor([range(SHARE_LENGTH)],
                                dtype=torch.long).to(device)
    attention_mask = torch.randn(
        (1, 1, SHARE_LENGTH, SHARE_LENGTH)).to(dtype).to(device)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/block_share_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)


def convert_block_unshare(layer_id):
    model = BlockCache(layer_id)
    hidden_states = torch.randn(
        (1, UNSHARE_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor([range(UNSHARE_LENGTH)],
                                dtype=torch.long).to(device)
    attention_mask = (torch.ones(
        (1, 1, UNSHARE_LENGTH,
         SHARE_LENGTH + UNSHARE_LENGTH)).to(dtype).to(device))
    past_k = (torch.randn(
        (1, SHARE_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device))
    past_v = (torch.randn(
        (1, SHARE_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device))

    torch.onnx.export(
        model,
        (hidden_states, position_ids, attention_mask, past_k, past_v),
        f"{folder}/block_unshare_{layer_id}.onnx",
        verbose=False,
        input_names=[
            "input_states", "position_ids", "attention_mask", "history_k",
            "history_v"
        ],
        output_names=["hidden_states", "past_k", "past_v"],
        do_constant_folding=True,
        opset_version=15,
    )


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
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).to(dtype).to(device)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head.pt')


def build_prompt(query):
    return f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'


def build_prompt_batch4(share, unshare0, unshare1, unshare2, unshare3):
    a = f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{share}'
    b = f'{unshare0}<|im_end|>\n<|im_start|>assistant\n'
    c = f'{unshare1}<|im_end|>\n<|im_start|>assistant\n'
    d = f'{unshare2}<|im_end|>\n<|im_start|>assistant\n'
    e = f'{unshare3}<|im_end|>\n<|im_start|>assistant\n'
    return a, b, c, d, e


def test_net_with_mask():
    embed = Embedding().to(device)
    blocks = [Block(i).to(device) for i in range(NUM_LAYERS)]
    block_kvs = [BlockCache(i).to(device) for i in range(NUM_LAYERS)]
    lm = LmHead()
    share_qr = """以下是周国平的散文《有所敬畏》
    在这个世界上，有的人信神，有的人不信，由此而区分为有神论者和无神论者、宗教徒和俗人。不过，这个区分并非很重要。还有一个比这重要得多的区分，便是有的人相信神圣，有的人不相信，人由此而分出了高尚和卑鄙。
一个人可以不信神，但不可以不相信神圣。是否相信上帝、佛、真主或别的什么主宰宇宙的神秘力量，往往取决于个人所隶属的民族传统、文化背景和个人的特殊经历，甚至取决于个人的某种神秘体验，这是勉强不得的。一个没有这些宗教信仰的人，仍然可能是一个善良的人。然而，倘若不相信人世间有任何神圣价值，百无禁忌，为所欲为，这样的人就与禽兽无异了。
相信神圣的人有所敬畏。在他的心目中，总有一些东西属于做人的根本，是亵渎不得的。他并不是害怕受到惩罚，而是不肯丧失基本的人格。不论他对人生怎样充满着欲求，他始终明白，一旦人格扫地，他在自己面前竟也失去了做人的自信和尊严，那么，一切欲求的满足都不能挽救他的人生的彻底失败。
相反，对于那些毫无敬畏之心的人来说，是不存在人格上的自我反省的。如果说“知耻近乎勇”，那么，这种人因为不知耻便显出一种卑怯的无赖相和残忍相。只要能够不受惩罚，他们可以在光天化日下干任何事，欺负、迫害乃至残杀无辜的弱者。盗匪之中，多这种愚昧兼无所敬畏之徒。一种消极的表现则是对他人生命的极端冷漠，见死不救，如今这类事既频频发生在众多路人旁观歹徒行凶的现场，也频频发生在号称治病救人实则草芥人命的某些医院里。类似行为每每使善良的人们不解，因为善良的人们无法相信，世上竟然真的会有这样丧失起码人性的人。在一个正常社会里，这种人总是极少数，并且会受到法律或正义力量的制裁。可是，当一个民族普遍丧失对神圣价值的信念时，这种人便可能相当多地滋生出来，成为触目惊心的颓败征兆。
赤裸裸的凶蛮和冷漠只是不知耻的粗糙形式，不知耻还有稍微精致一些的形式。有的人有很高的文化程度，仍然可能毫无敬畏之心。他可以玩弄真心爱他的女人，背叛诚恳待他的朋友，然后装出一副无辜的面孔。他的足迹所到之处，再神圣的东西也敢践踏，再美好的东西也敢毁坏，而且内心没有丝毫不安。不论他的头脑里有多少知识，他的心是蒙昧的，真理之光到不了那里。这样的人有再多的艳遇，也没有能力真正爱一回，交再多的哥们，也体味不了友谊的纯正，获取再多的名声，也不知什么是光荣。我对此深信不疑：不相信神圣的人，必被世上一切神圣的事物所抛弃。
"""
    unshare_q1 = "请总结这篇文章内容"
    unshare_q0 = "请介绍该作者"
    unshare_q2 = "请评价这篇文章写得怎么样"
    unshare_q3 = "请说明该文章的写作背景"

    qs, q0, q1, q2, q3 = build_prompt_batch4(share_qr, unshare_q0, unshare_q1,
                                             unshare_q2, unshare_q3)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    ids_qs = tokenizer.encode(qs)
    #ids_q0 = tokenizer.encode(q0)
    ids_q0 = [
        14880, 100157, 75882, 57421, 198, 151645, 198, 151644, 77091, 198
    ]
    ids_q1 = tokenizer.encode(q1)
    ids_q2 = tokenizer.encode(q2)
    ids_q3 = tokenizer.encode(q3)
    print(ids_qs)
    print(ids_q0)
    print("===========")

    ## prefill ids_qs share part
    qs_len = len(ids_qs)
    ids = ids_qs + (SHARE_LENGTH - qs_len) * [0]
    input_ids = torch.tensor(ids).view(SHARE_LENGTH).to(device)
    out = embed(input_ids).view(1, SHARE_LENGTH, HIDDEN_SIZE)
    position_ids = list(range(qs_len)) + (SHARE_LENGTH - qs_len) * [0]
    position_ids = torch.tensor([position_ids]).to(device)
    attention_mask = torch.ones(
        (SHARE_LENGTH, SHARE_LENGTH)).float() * -10000.0
    for i in range(qs_len):
        for j in range(qs_len):
            if j <= i:
                attention_mask[i][j] = 0.0
    attention_mask = attention_mask.view(1, 1, SHARE_LENGTH,
                                         SHARE_LENGTH).to(device)
    k_cache = []
    v_cache = []
    for i in range(NUM_LAYERS):
        out[:, qs_len:] = 0
        out, k, v = blocks[i](out.to(dtype), position_ids, attention_mask)
        k[:, qs_len:, :, :] = 0
        v[:, qs_len:, :, :] = 0
        k_cache.append(k)
        v_cache.append(v)

    ## prefill ids_q0
    q0_len = len(ids_q0)
    ids = ids_q0 + (UNSHARE_LENGTH - q0_len) * [0]
    input_ids = torch.tensor(ids).view(UNSHARE_LENGTH).to(device)
    out = embed(input_ids).view(1, UNSHARE_LENGTH, HIDDEN_SIZE)
    out[:, q0_len:, :] = 0.0
    position_ids = list(range(
        qs_len, qs_len + q0_len)) + (UNSHARE_LENGTH - q0_len) * [0]
    position_ids = torch.tensor([position_ids]).to(device)
    attention_mask = torch.ones(
        (UNSHARE_LENGTH, SEQ_LENGTH)).float() * -10000.0
    for i in range(q0_len):
        for j in range(qs_len):
            attention_mask[i][j] = 0.0  #if j <= i:
    for i in range(q0_len):
        for j in range(q0_len):
            attention_mask[i][SHARE_LENGTH + j] = 0.0
    attention_mask = attention_mask.view(1, 1, UNSHARE_LENGTH,
                                         SEQ_LENGTH).to(device)
    k0_cache = []
    v0_cache = []
    data = {}
    for i in range(NUM_LAYERS):
        out[:, q0_len:, :] = 0
        out, k, v = block_kvs[i](out.to(dtype), position_ids, attention_mask,
                                 k_cache[i].to(dtype), v_cache[i].to(dtype))
        k_zeros = torch.zeros(1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS,
                              HEAD_DIM).to(device)
        v_zeros = torch.zeros(1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS,
                              HEAD_DIM).to(device)
        k_zeros[:, :qs_len, :, :] = k_cache[i][:, :qs_len, :, :]
        k_zeros[:, qs_len:qs_len + q0_len, :, :] = k[:, :q0_len, :, :]
        v_zeros[:, :qs_len, :, :] = v_cache[i][:, :qs_len, :, :]
        v_zeros[:, qs_len:qs_len + q0_len, :, :] = v[:, :q0_len, :, :]
        data[f"k{i}"]=k_zeros.cpu()
        data[f"v{i}"]=v_zeros.cpu()  
        k0_cache.append(k_zeros)
        v0_cache.append(v_zeros)
    np.savez("first_kv_out.npz", **data)
    out = out[:, q0_len - 1:q0_len].view(1, 1, HIDDEN_SIZE)
    token = lm(out.to(dtype)).view(1)
    out_ids = [int(token)]
    word = tokenizer.decode([int(token)])
    print(word, end="")
    # decode ids_q0
    token_len = qs_len + q0_len
    while int(token) != tokenizer.eos_token_id and token_len < SEQ_LENGTH:
        token_len += 1
        input_ids = torch.tensor([token]).to(device)
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]]).to(device)
        attention_mask = torch.zeros(
            (1, 1, 1, SEQ_LENGTH + 1)).float().to(device)
        attention_mask[:, :, :, token_len:SEQ_LENGTH] = -10000.0
        for i in range(NUM_LAYERS):
            out, k, v = block_kvs[i](out.to(dtype), position_ids,
                                     attention_mask, k0_cache[i].to(dtype),
                                     v0_cache[i].to(dtype))
            k0_cache[i][:, token_len:token_len + 1] = k
            v0_cache[i][:, token_len:token_len + 1] = v
        token = lm(out.to(dtype)).view(1)
        out_ids.append(int(token))
        word = tokenizer.decode([int(token)])
        print(word, end="")
    print("\noutput_ids:{}".format(out_ids))


test_net_with_mask()

# create folder to store onnx
# if not os.path.exists(folder):
#     os.makedirs(folder)

# # export models
# print('Convert block & block_cache')
# for i in tqdm(range(NUM_LAYERS)):
#     convert_block(i)
#     convert_block_cache(i)

# print('Convert embedding')
# convert_embedding()

# print('Convert lm_head')
# convert_lm_head()
