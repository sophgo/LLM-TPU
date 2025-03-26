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
import copy
import torch
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
parser.add_argument('-s', '--seq_length', type=int, default=512, help="sequence length")
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")

args = parser.parse_args()

model_path = args.model_path
folder = "./tmp/onnx"

device = torch.device(args.device)
if args.device == "cpu":
    dtype = torch.float
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    dtype = torch.bfloat16

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
origin_model = AutoModelForCausalLM.from_config(
    config, trust_remote_code=True, attn_implementation='eager',
    torch_dtype=dtype).eval()

for param in origin_model.parameters():
    param.requires_grad = False

config = origin_model.config
transformer = origin_model.model
layers = transformer.layers

DEVICE_NUM = 8
SEQ_LENGTH = args.seq_length
NUM_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
NUM_KEY_VALUE_HEADS = config.num_key_value_heads
NOPE_HEAD_DIM = config.qk_nope_head_dim
ROPE_HEAD_DIM = config.qk_rope_head_dim
HEAD_DIM = NOPE_HEAD_DIM + ROPE_HEAD_DIM
VOCAB_SIZE = config.vocab_size

print(f'Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n')

class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        out = transformer.embed_tokens(input_ids)
        return out.float()


class Attention(torch.nn.Module):

    def __init__(self, layer_id, device_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.layer = copy.deepcopy(layers[layer_id])
        self.split_weights(self.layer.self_attn, device_id, DEVICE_NUM)

        self.scale_by_device = 1 / DEVICE_NUM
    
    def split_weights(self, module, device_id, device_num):
        module.q_b_proj.weight = torch.nn.Parameter(module.q_b_proj.weight.chunk(device_num, dim=0)[device_id])
        module.kv_b_proj.weight = torch.nn.Parameter(module.kv_b_proj.weight.chunk(device_num, dim=0)[device_id])

        module.o_proj.weight = torch.nn.Parameter(module.o_proj.weight.chunk(device_num, dim=1)[device_id])
        module.num_heads = module.num_heads // device_num

    def forward(self, hidden_states, position_ids, attention_mask):
        residual = hidden_states * self.scale_by_device

        hidden_states = self.layer.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _, present_kv = self.layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=None,
            use_cache=True
        )
        present_key, present_value = present_kv[0], present_kv[1]
        hidden_states = residual + hidden_states
        return hidden_states.float(), present_key.float(), present_value.float()
    

class AttentionCache(torch.nn.Module):

    def __init__(self, layer_id, device_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.layer = copy.deepcopy(layers[layer_id])
        self.split_weights(self.layer.self_attn, device_id, DEVICE_NUM)

        self.scale_by_device = 1 / DEVICE_NUM
    
    def split_weights(self, module, device_id, device_num):
        module.q_b_proj.weight = torch.nn.Parameter(module.q_b_proj.weight.chunk(device_num, dim=0)[device_id])
        module.kv_b_proj.weight = torch.nn.Parameter(module.kv_b_proj.weight.chunk(device_num, dim=0)[device_id])

        module.o_proj.weight = torch.nn.Parameter(module.o_proj.weight.chunk(device_num, dim=1)[device_id])
        module.num_heads = module.num_heads // device_num

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        residual = hidden_states * self.scale_by_device

        hidden_states = self.layer.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _, present_kv = self.layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=(past_k, past_v),
            output_attentions=None,
            use_cache=False
        )
        present_key, present_value = present_kv[0], present_kv[1]
        hidden_states = residual + hidden_states
        return hidden_states.float(), present_key.float(), present_value.float()



class MLP(torch.nn.Module):

    def __init__(self, layer_id, device_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.layer = copy.deepcopy(layers[layer_id])
        self.split_weights(self.layer.mlp, device_id, DEVICE_NUM)

        self.scale_by_device = 1 / DEVICE_NUM
    
    def split_weights(self, module, device_id, device_num):
        module.gate_proj.weight = torch.nn.Parameter(module.gate_proj.weight.chunk(device_num, dim=0)[device_id])
        module.up_proj.weight = torch.nn.Parameter(module.up_proj.weight.chunk(device_num, dim=0)[device_id])

        module.down_proj.weight = torch.nn.Parameter(module.down_proj.weight.chunk(device_num, dim=1)[device_id])

    def forward(self, hidden_states):
        residual = hidden_states * self.scale_by_device
        hidden_states = self.layer.post_attention_layernorm(hidden_states)
        hidden_states = self.layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    


class SharedMOE(torch.nn.Module):

    def __init__(self, layer_id, device_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.layer = copy.deepcopy(layers[layer_id])
        self.split_weights(self.layer.mlp, device_id, DEVICE_NUM)

        self.scale_by_device = 1 / DEVICE_NUM
    
    def split_weights(self, module, device_id, device_num):
        gate = module.gate
        gate.weight = torch.nn.Parameter(gate.weight.chunk(device_num, dim=0)[device_id])

        shared_experts = module.shared_experts
        shared_experts.gate_proj.weight = torch.nn.Parameter(shared_experts.gate_proj.weight.chunk(device_num, dim=0)[device_id])
        shared_experts.up_proj.weight = torch.nn.Parameter(shared_experts.up_proj.weight.chunk(device_num, dim=0)[device_id])

        shared_experts.down_proj.weight = torch.nn.Parameter(shared_experts.down_proj.weight.chunk(device_num, dim=1)[device_id])


    def forward(self, hidden_states):
        residual = hidden_states * self.scale_by_device
        hidden_states = self.layer.post_attention_layernorm(hidden_states)
        logits = F.linear(hidden_states, self.layer.mlp.gate.weight)

        out_hidden_states = self.layer.mlp.shared_experts(hidden_states)
        out_hidden_states = residual + out_hidden_states
        return out_hidden_states, logits


class MOE(torch.nn.Module):

    def __init__(self, layer_id, device_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.layer = copy.deepcopy(layers[layer_id])
        self.split_weights(self.layer.mlp, device_id, DEVICE_NUM)

        self.scale_by_device = 1 / DEVICE_NUM
    
    def split_weights(self, module, device_id, device_num):
        for i in range(len(module.experts)):
            expert = module.experts[i]
            expert.gate_proj.weight = torch.nn.Parameter(expert.gate_proj.weight.chunk(device_num, dim=0)[device_id])
            expert.up_proj.weight = torch.nn.Parameter(expert.up_proj.weight.chunk(device_num, dim=0)[device_id])

            expert.down_proj.weight = torch.nn.Parameter(expert.down_proj.weight.chunk(device_num, dim=1)[device_id])

    def forward(self, hidden_states, token_index, re_index, topk_weight, shared_out):
        # token_index [160, 60]
        # re_index [512 * 6]
        # topk_weight [6,1,1]
        # shared_out [1, 512, 5120]
        out = []
        for i in range(160):
            tmp_hidden_states = hidden_states[token_index[i]].unsqueeze(0)
            out_hidden_states = self.layer.mlp.experts[i](tmp_hidden_states)
            out.append(out_hidden_states)
        out_hidden_states = torch.cat(out, dim=1)
        out_hidden_states = out_hidden_states[:,re_index]
        out_hidden_states = (out_hidden_states.reshape(6, SEQ_LENGTH, HIDDEN_SIZE) * topk_weight).sum(dim=0, keepdim=True) + shared_out
        return out_hidden_states


class MOECache(torch.nn.Module):

    def __init__(self, layer_id, expert_id=0, device_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.layer = copy.deepcopy(layers[layer_id])
        self.split_weights(self.layer.mlp, expert_id, device_id, DEVICE_NUM)

        self.cur_expert = self.layer.mlp.experts[expert_id]
        self.scale_by_device = 1 / DEVICE_NUM
    
    def split_weights(self, module, expert_id, device_id, device_num):
        expert = module.experts[expert_id]
        expert.gate_proj.weight = torch.nn.Parameter(expert.gate_proj.weight.chunk(device_num, dim=0)[device_id])
        expert.up_proj.weight = torch.nn.Parameter(expert.up_proj.weight.chunk(device_num, dim=0)[device_id])

        expert.down_proj.weight = torch.nn.Parameter(expert.down_proj.weight.chunk(device_num, dim=1)[device_id])

    def forward(self, hidden_states, expert_weight, shared_out):
        out_hidden_states = self.cur_expert(hidden_states) * expert_weight + shared_out
        return out_hidden_states

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


def convert_attention(layer_id, device_id):
    model = Attention(layer_id, device_id)
    hidden_states = torch.randn(
        (1, SEQ_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor(
        [range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.randn(
        (1, 1, SEQ_LENGTH, SEQ_LENGTH)).to(dtype).to(device)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/attention_layer{layer_id}_dev{device_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)


def convert_attention_cache(layer_id, device_id):
    model = AttentionCache(layer_id, device_id)
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device)
    attention_mask = torch.ones(
        (1, 1, 1, SEQ_LENGTH + 1)).to(dtype).to(device)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS // DEVICE_NUM, HEAD_DIM)).to(dtype).to(device)
    past_v = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS // DEVICE_NUM, NOPE_HEAD_DIM)).to(dtype).to(device)

    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask, past_k, past_v),
        f'{folder}/attention_cache_layer{layer_id}_dev{device_id}.onnx',
        verbose=False,
        input_names=[
            'input_states', 'position_ids', 'attention_mask', 'history_k',
            'history_v'
        ],
        output_names=['hidden_states', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=15)
    

def convert_mlp(layer_id, device_id):
    model = MLP(layer_id, device_id)
    hidden_states = torch.randn(
        (1, SEQ_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)

    torch.onnx.export(
        model, (hidden_states),
        f'{folder}/mlp_layer{layer_id}_dev{device_id}.onnx',
        verbose=False,
        input_names=['input_states'],
        output_names=['hidden_states'],
        do_constant_folding=True,
        opset_version=15)


def convert_mlp_cache(layer_id, device_id):
    model = MLP(layer_id, device_id)
    hidden_states = torch.randn(
        (1, 1, HIDDEN_SIZE)).to(dtype).to(device)

    torch.onnx.export(
        model, (hidden_states),
        f'{folder}/mlp_cache_layer{layer_id}_dev{device_id}.onnx',
        verbose=False,
        input_names=['input_states'],
        output_names=['hidden_states'],
        do_constant_folding=True,
        opset_version=15)


def convert_shared_moe(layer_id, device_id):
    model = SharedMOE(layer_id, device_id)
    hidden_states = torch.randn(
        (1, SEQ_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)

    torch.onnx.export(
        model, (hidden_states),
        f'{folder}/shared_moe_layer{layer_id}_dev{device_id}.onnx',
        verbose=False,
        input_names=['input_states', "top_index"],
        output_names=['hidden_states'],
        do_constant_folding=True,
        opset_version=15)
    

def convert_shared_moe_cache(layer_id, device_id):
    model = SharedMOE(layer_id, device_id)
    hidden_states = torch.randn(
        (1, 1, HIDDEN_SIZE)).to(dtype).to(device)

    torch.onnx.export(
        model, (hidden_states),
        f'{folder}/shared_moe_cache_layer{layer_id}_dev{device_id}.onnx',
        verbose=False,
        input_names=['input_states'],
        output_names=['hidden_states'],
        do_constant_folding=True,
        opset_version=15)

def convert_moe(layer_id, device_id):
    model = MOE(layer_id, device_id)
    hidden_states = torch.randn(
        (SEQ_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
    token_index = torch.LongTensor(list(range(2))*80*60).reshape(160,60)
    re_index = torch.LongTensor(list(range(6))*512)
    topk_weight = torch.randn(6).reshape(6,1,1)
    shared_out = torch.randn(1, SEQ_LENGTH, HIDDEN_SIZE)

    torch.onnx.export(
        model, (hidden_states, token_index, re_index, topk_weight, shared_out),
        f'{folder}/moe_layer{layer_id}_dev{device_id}.onnx',
        verbose=False,
        input_names=['input_states', "token_index", "re_index", "topk_weight", "shared_out"],
        output_names=['hidden_states'],
        do_constant_folding=True,
        opset_version=15)
    

def convert_moe_cache(layer_id, expert_id, device_id):
    model = MOECache(layer_id, device_id)
    hidden_states = torch.randn(
        (1, 1, HIDDEN_SIZE)).to(dtype).to(device)
    expert_weight = torch.randn(1).reshape(1,1,1)
    shared_out = torch.randn(1, 1, HIDDEN_SIZE)

    torch.onnx.export(
        model, (hidden_states, expert_weight, shared_out),
        f'{folder}/moe_cache_expert{expert_id}_layer{layer_id}_dev{device_id}.onnx',
        verbose=False,
        input_names=['input_states', 'expert_weight', 'shared_out'],
        output_names=['hidden_states'],
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
    
def test_deepseek_v2_net():
    # 初始化模块
    embed = Embedding().to(device)
    blocks = [Attention(layer_id, 0).to(device) for layer_id in range(NUM_LAYERS)]
    block_caches = [AttentionCache(layer_id, 0).to(device) for layer_id in range(NUM_LAYERS)]
    mlps = [MLP(layer_id, 0).to(device) if layer_id < 1 else 
            SharedMOE(layer_id, 0).to(device) for layer_id in range(NUM_LAYERS)]  # 假设前2层是MLP，后面是MoE
    lm_head = LmHead().to(device)
    greedy_head = GreedyHead().to(device)

    # 测试数据准备
    tokenizer_path = "/workspace/models/DeepSeek-V2.5-1210-quantized.w4a16/"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    messages = [
        {"role": "user", "content": "Write a piece of quicksort code in C++"}
    ]
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    input_ids = input_tensor.squeeze(0).tolist()
    token_len = len(input_ids)
    seq_length = SEQ_LENGTH

    # 首轮推理（完整序列）
    # 准备输入
    input_ids = input_ids + [0] * (seq_length - token_len)
    input_tensor = torch.tensor(input_ids, device=device)
    
    # 嵌入层
    hidden_states = embed(input_tensor).view(1, seq_length, HIDDEN_SIZE)
    
    # 位置编码
    position_ids = torch.tensor([list(range(token_len)) + [0]*(seq_length - token_len)], device=device)
    
    # 注意力掩码（下三角）
    attention_mask = torch.full((seq_length, seq_length), -float('inf'), device=device)
    for i in range(token_len):
        attention_mask[i, :i+1] = 0
    attention_mask = attention_mask.view(1, 1, seq_length, seq_length)

    # 逐层处理
    kv_caches = []
    for layer_id in range(NUM_LAYERS):
        # 注意力层
        hidden_states, k, v = blocks[layer_id](
            hidden_states.to(dtype),
            position_ids,
            attention_mask
        )
        kv_caches.append((k, v))
        
        # FFN层
        if layer_id < 1:  # MLP
            hidden_states = mlps[layer_id](hidden_states.to(dtype))
        else:  # MoE
            # Shared expert处理
            shared_out = mlps[layer_id](hidden_states.to(dtype))
            # 假设已经通过门控获得topk信息
            topk_idx = torch.randint(0, 160, (1, seq_length, 6))  # 模拟topk索引
            topk_weight = torch.rand(1, seq_length, 6, 1, 1)     # 模拟topk权重
            # MoE处理（需要实际的门控逻辑）
            hidden_states = shared_out  # 此处简化处理

    # 生成首个token
    logits = lm_head(hidden_states[:, token_len-1, :].unsqueeze(1))
    next_token = greedy_head(logits)

    # 自回归生成
    max_new_tokens = 10
    for _ in range(max_new_tokens):
        # 准备输入（单token）
        input_tensor = next_token
        position_ids = torch.tensor([[token_len]], device=device)
        
        # 嵌入层
        hidden_states = embed(input_tensor).view(1, 1, HIDDEN_SIZE)
        
        # 注意力掩码（扩展缓存）
        attention_mask = torch.zeros(1, 1, 1, seq_length+1, device=device)
        attention_mask[:, :, :, token_len+1:] = -float('inf')

        # 逐层处理
        new_kv_caches = []
        for layer_id in range(NUM_LAYERS):
            # 读取历史KV
            past_k, past_v = kv_caches[layer_id]
            
            # 注意力层（带缓存）
            hidden_states, new_k, new_v = block_caches[layer_id](
                hidden_states.to(dtype),
                position_ids,
                attention_mask,
                past_k[:, :token_len],
                past_v[:, :token_len]
            )
            
            # 更新KV缓存
            updated_k = torch.cat([past_k, new_k], dim=1)
            updated_v = torch.cat([past_v, new_v], dim=1)
            new_kv_caches.append((updated_k, updated_v))
            
            # FFN层
            if layer_id < 1:
                hidden_states = mlps[layer_id](hidden_states.to(dtype))
            else:
                # 简化处理MoE
                hidden_states = mlps[layer_id](hidden_states.to(dtype))

        kv_caches = new_kv_caches
        
        # 生成下一个token
        logits = lm_head(hidden_states)
        next_token = greedy_head(logits)
        token_len += 1

        # 终止条件检查
        if next_token.item() == tokenizer.eos_token_id:
            break


def test_deepseek_v2_net():
    # 初始化模块
    embed = Embedding().to(device)
    attentions = [Attention(layer_id, 0).to(device) for layer_id in range(1)]
    attention_caches = [AttentionCache(layer_id, 0).to(device) for layer_id in range(1)]
    
    # MoE相关模块初始化
    shared_moes = [SharedMOE(layer_id, 0).to(device) for layer_id in range(1, NUM_LAYERS)]
    shared_moe_caches = [SharedMOE(layer_id, 0).to(device) for layer_id in range(1, NUM_LAYERS)]
    moes = [[MOE(layer_id, 0).to(device) for _ in range(160)] for layer_id in range(1, NUM_LAYERS)]
    moe_caches = [[[MOECache(layer_id, expert_id, 0).to(device) 
                   for expert_id in range(160)] for _ in range(DEVICE_NUM)] 
                 for layer_id in range(1, NUM_LAYERS)]
    
    lm_head = LmHead().to(device)
    greedy_head = GreedyHead().to(device)

    # 测试数据准备
    tokenizer_path = "/workspace/models/DeepSeek-V2.5-1210-quantized.w4a16/"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    messages = [{"role": "user", "content": "Write a piece of quicksort code in C++"}]
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    input_ids = input_tensor.squeeze(0).tolist()
    token_len = len(input_ids)
    seq_length = SEQ_LENGTH

    # 首轮推理（完整序列）
    # 准备输入
    padded_ids = input_ids + [0] * (seq_length - token_len)
    input_tensor = torch.tensor(padded_ids, device=device)
    
    # 嵌入层
    hidden_states = embed(input_tensor).view(1, seq_length, HIDDEN_SIZE)
    
    # 位置编码
    position_ids = torch.tensor([list(range(token_len)) + [0]*(seq_length - token_len)], device=device)
    
    # 注意力掩码（下三角）
    attention_mask = torch.full((seq_length, seq_length), -10000, device=device)
    for i in range(token_len):
        attention_mask[i, :i+1] = 0
    attention_mask = attention_mask.view(1, 1, seq_length, seq_length)

    # 逐层处理
    kv_caches = []
    for layer_id in range(NUM_LAYERS):
        # 注意力层
        breakpoint()
        hidden_states, k, v = attentions[layer_id](
            hidden_states.to(dtype),
            position_ids,
            attention_mask
        )
        kv_caches.append((k, v))
        
        # MoE处理
        # Shared expert前向
        shared_out, gate_logits = shared_moes[layer_id](hidden_states.to(dtype))
        
        # 门控计算 (示例实现)
        topk_val, topk_idx = torch.topk(gate_logits, k=6, dim=-1)  # 假设选择top6专家
        topk_weight = torch.softmax(topk_val.float(), dim=-1).to(dtype)
        
        # 重组索引
        token_index = topk_idx.view(-1).argsort().view(160, 60)  # 模拟索引重组
        re_index = torch.arange(seq_length*6, device=device)    # 模拟重排索引
        
        # MoE专家计算
        hidden_states = moes[layer_id][0](  # 示例使用第一个设备上的专家
            hidden_states.to(dtype),
            token_index,
            re_index,
            topk_weight.view(6,1,1),
            shared_out
        )

    # 生成首个token
    logits = lm_head(hidden_states[:, token_len-1, :].unsqueeze(1))
    next_token = greedy_head(logits)

    # 自回归生成
    max_new_tokens = 10
    for _ in tqdm(range(max_new_tokens)):
        # 准备输入
        input_tensor = next_token
        position_ids = torch.tensor([[token_len]], device=device)
        
        # 嵌入层
        hidden_states = embed(input_tensor).view(1, 1, HIDDEN_SIZE)
        
        # 注意力掩码（扩展缓存）
        attention_mask = torch.zeros(1, 1, 1, seq_length+1, device=device)
        attention_mask[:, :, :, token_len+1:] = -float('inf')

        # 逐层处理
        new_kv_caches = []
        for layer_id in range(NUM_LAYERS):
            # 注意力层（带缓存）
            past_k, past_v = kv_caches[layer_id]
            hidden_states, new_k, new_v = attention_caches[layer_id](
                hidden_states.to(dtype),
                position_ids,
                attention_mask,
                past_k[:, :token_len],
                past_v[:, :token_len]
            )
            new_kv_caches.append((torch.cat([past_k, new_k], dim=1),
                                torch.cat([past_v, new_v], dim=1)))
            
            # MoE缓存处理
            # Shared expert前向
            shared_out, gate_logits = shared_moe_caches[layer_id](hidden_states.to(dtype))
            
            # 门控计算（单token）
            topk_val, topk_idx = torch.topk(gate_logits[:, -1:], k=6, dim=-1)
            topk_weight = torch.softmax(topk_val.float(), dim=-1).to(dtype)
            
            # 使用第一个设备的专家计算
            expert_outputs = []
            for expert_id in range(160):
                if expert_id in topk_idx:
                    expert_out = moe_caches[layer_id][0][expert_id](
                        hidden_states.to(dtype),
                        topk_weight[0, 0, expert_id].view(1,1,1),
                        shared_out
                    )
                    expert_outputs.append(expert_out)
            hidden_states = sum(expert_outputs) / len(expert_outputs)

        kv_caches = new_kv_caches
        
        # 生成下一个token
        logits = lm_head(hidden_states)
        next_token = greedy_head(logits)
        token_len += 1

        if next_token.item() == tokenizer.eos_token_id:
            break

# 运行测试
test_deepseek_v2_net()

# test_net_with_mask()

# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# export models
print('Convert block & block_cache')
for device_id in range(1):
    for layer_id in tqdm(range(1)):
        convert_attention(layer_id, device_id)
        convert_attention_cache(layer_id, device_id)
        convert_mlp(layer_id, device_id)
        convert_mlp_cache(layer_id, device_id)


for device_id in range(1):
    for layer_id in tqdm(range(1, NUM_LAYERS)):
        convert_attention(layer_id, device_id)
        convert_attention_cache(layer_id, device_id)

        convert_shared_moe(layer_id, device_id)
        convert_shared_moe_cache(layer_id, device_id)
        convert_moe(layer_id, device_id)
        for expert_id in range(160):
            convert_moe_cache(layer_id, expert_id, device_id)

# print('Convert embedding')
# convert_embedding()

# print('Convert lm_head')
# convert_lm_head()
# convert_greedy_head()
# convert_penalty_sample_head()
