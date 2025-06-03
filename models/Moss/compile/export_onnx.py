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
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from tqdm import tqdm
import pdb
import inspect

parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
parser.add_argument('-s', '--seq_length', type=int, default=512, help="sequence length")


args = parser.parse_args()

model_path = args.model_path
folder = "./tmp/onnx"

dtype = torch.bfloat16

# if not os.path.exists(model_path):
#     model_path = snapshot_download(model_path)
config = AutoConfig.from_pretrained("moss-moon-003-sft", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("moss-moon-003-sft", trust_remote_code=True)
with init_empty_weights():
    origin_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
origin_model.tie_weights()
origin_model = load_checkpoint_and_dispatch(origin_model, model_path, device_map="auto", no_split_module_classes=["MossBlock"], dtype=torch.float16)


for param in origin_model.parameters():
    param.requires_grad = False

config = origin_model.config
# transformer = origin_model.model
# layers = transformer.layers
transformer = origin_model.transformer
layers = transformer.h

device="cuda:0"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

SEQ_LENGTH = args.seq_length
NUM_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
# NUM_KEY_VALUE_HEADS = config.num_key_value_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
VOCAB_SIZE = config.vocab_size

print(f'Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n')

class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        out = transformer.wte(input_ids)
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
        # pdb.set_trace()
        hidden_states, past_kv = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=(past_k, past_v),
            #past_key_values=(past_k, past_v),
            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = transformer.ln_f(hidden_states)
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


    
# refs:https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py


def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn(
        (1, SEQ_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor(
        [range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.randn(
        (1, 1, SEQ_LENGTH, SEQ_LENGTH)).to(dtype).to(device)

    #pdb.set_trace()
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
    past_k = torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(dtype).to(device)
    past_v = torch.randn((1, SEQ_LENGTH, NUM_ATTENTION_HEADS, HEAD_DIM)).to(dtype).to(device)

    # pdb.set_trace()
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
    # with torch.jit.optimized_execution(True):  # 开启优化执行
        # module = torch.jit.script(model.forward, input_ids)
    torch.jit.save(module, f'{folder}/embedding.pt')
    

def convert_lm_head():
    model = LmHead()
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).to(dtype).to(device)
    # print(hidden_states.requires_grad)  # 应该是 False
    # print("hidden_states:", hidden_states)
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
    return f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'

# def test_net_with_mask():
#     embed = Embedding().to(device)  # 词嵌入层i
#     blocks = [Block(i).to(device) for i in range(NUM_LAYERS)]  # 网络的每一层
#     block_kvs = [BlockCache(i).to(device) for i in range(NUM_LAYERS)]  # Cache 用于存储每层的 key 和 value
#     ids = tokenizer.encode('你好')  # 输入句子的编码
#     query = '你好'
#     print(query)
#     promt = build_prompt(query)
#     ids = tokenizer.encode(promt)
#     print("input ids:{}".format(ids))
    
#     token_len = len(ids)
#     ids = ids + (SEQ_LENGTH - token_len) * [0]  # 填充输入
#     input_ids = torch.tensor(ids).view(SEQ_LENGTH).to(device)  # 转化为 tensor
#     out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)  # 经过嵌入层得到隐藏状态
#     position_ids = list(range(token_len)) + (SEQ_LENGTH - token_len) * [0]
#     position_ids = torch.tensor([position_ids]).to(device)
    
#     # 初始化 attention_mask
#     attention_mask = torch.ones((SEQ_LENGTH, SEQ_LENGTH))
#     for i in range(token_len):
#         for j in range(token_len):
#             if j <= i:
#                 attention_mask[i][j] = 0
#     attention_mask = attention_mask.view(1, 1, SEQ_LENGTH, SEQ_LENGTH).to(device)
    
#     # 存储每层的 key 和 value
#     k_cache = []
#     v_cache = []

#     # 循环遍历每一层进行推理
#     for i in range(NUM_LAYERS):
#         out[:, token_len:] = 0  # 确保 padding 部分不参与推理
#         out, k, v = blocks[i](out.to(dtype), position_ids, attention_mask)  # 前向传播
#         k_cache.append(k)  # 缓存 k
#         v_cache.append(v)  # 缓存 v
        
#         # 输出每层的结果（用于验证模型的正确性）
#         print(f"Layer {i} output: {out.shape}")
#         # 可以在这里插入一些验证代码，如对比输出的值和预期结果

#     out = out[:, token_len - 1:token_len].view(1, 1, HIDDEN_SIZE)  # 从输出中截取最后一个 token
#     lm = LmHead()  # 语言模型头
#     token = lm(out.to(dtype)).view(1)  # 获取输出 token
#     out_ids = [int(token)]  # 存储输出的 token id
#     word = tokenizer._convert_id_to_token(int(token[0]))  # 转换为对应的词
#     print(word, end="")

#     # 循环生成更多 token，直到达到最大长度或输出结束标志
#     while token > 2 and token_len < 64:
#         token_len += 1
#         input_ids = torch.tensor([token]).to(device)  # 使用上一个 token 作为输入
#         out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
#         position_ids = torch.tensor([[token_len - 1]]).to(device)
        
#         # 更新 attention_mask
#         attention_mask = torch.zeros((1, 1, 1, SEQ_LENGTH + 1)).float().to(device)
#         attention_mask[:, :, :, SEQ_LENGTH + 1 - token_len:] = 0
        
#         # 逐层推理
#         for i in range(NUM_LAYERS):
#             out, k, v = block_kvs[i](out.to(dtype), position_ids, attention_mask, k_cache[i].to(dtype), v_cache[i].to(dtype))
#             k_cache[i][:SEQ_LENGTH - token_len] = 0  # 清空过期的缓存
#             v_cache[i][:SEQ_LENGTH - token_len] = 0

#         # 预测下一个 token
#         token = lm(out.to(dtype)).view(1)
#         out_ids.append(int(token))  # 存储生成的 token id
#         word = tokenizer._convert_id_to_token(int(token[0]))  # 转换为对应的词
#         print(word, end="")
    
#     print("\noutput_ids:{}".format(out_ids))  # 输出生成的 token id 序列



def test_net_with_mask():
    embed = Embedding().to(device)
    blocks = [Block(i).to(device) for i in range(20)]
    block_kvs = [BlockCache(i).to(device) for i in range(20)]
    query = """tell me something about moss in ten words"""
    print(query)
    promt = build_prompt(query)
    import numpy as np
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
    attention_mask = attention_mask.view(
        1, 1, SEQ_LENGTH, SEQ_LENGTH).to(device)
    k_cache = []
    v_cache = []
    for i in range(20):
        # breakpoint()
        out[:,token_len:] = 0
        out, k, v = blocks[i](out.to(dtype), position_ids, attention_mask)
        # print("----key shape----",k.shape)
        # k[:, SEQ_LENGTH - token_len:] = k[:, :token_len]
        # v[:, SEQ_LENGTH - token_len:] = v[:, :token_len]
        # k[:, :SEQ_LENGTH - token_len] = 0
        # v[:, :SEQ_LENGTH - token_len] = 0
        # np.save(f'torch_block_{i}.npy', out.float().cpu().numpy())
        k_cache.append(k)
        # print("----k_cache shape----",k_cache[i].shape)
        v_cache.append(v)
    out = out[:, token_len - 1:token_len].view(1, 1, HIDDEN_SIZE)
    lm = LmHead()
    greedy_head = GreedyHead()
    # np.save(f'torch_lm_head_{i}.npy', lm(out.to(dtype)).float().cpu().numpy())
    token = greedy_head(lm(out.to(dtype))).view(1)
    out_ids = [int(token)]
    word = tokenizer.decode([int(token)])
    print(word, end="")
    while int(token) != tokenizer.eos_token_id and token_len < ori_token_len + 10:
        token_len += 1
        input_ids = torch.tensor([token]).to(device)
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]]).to(device)
        attention_mask = torch.zeros((1, 1, 1, SEQ_LENGTH + 1)).float().to(device)
        attention_mask[:, :, :, token_len:SEQ_LENGTH] = -10000.0
        for i in range(20):
            out, k, v = block_kvs[i](out.to(dtype), position_ids, attention_mask, k_cache[i].to(dtype), v_cache[i].to(dtype))
            # print("----k_cache[i][:,token_len:token_len+1] shape----",k_cache[i][:,token_len:token_len+1].shape)
            k = k[:, :, -1:, :]  
            v = v[:, :, -1:, :] 
            # print("----key shape----",k.shape)
            k_cache[i][:,:,token_len:token_len+1] = k
            v_cache[i][:,:,token_len:token_len+1] = v
        token = greedy_head(lm(out.to(dtype))).view(1)
        out_ids.append(int(token))
        word = tokenizer.decode([int(token)])
        print(word, end="")
    print("\noutput_ids:{}".format(out_ids))



def test_first_layer_only():
    embed = Embedding().to(device)  # 词嵌入层
    block = Block(33).to(device)  # 只加载第一层模型
    block_kv = BlockCache(33).to(device)  # 只加载第一层缓存
    ids = tokenizer.encode('你好')  # 输入句子的编码
    query = '你好'
    print(query)
    promt = build_prompt(query)
    ids = tokenizer.encode(promt)
    print("input ids:{}".format(ids))
    
    token_len = len(ids)
    ori_token_len = token_len
    ids = ids + (SEQ_LENGTH - token_len) * [0]  # 填充输入
    input_ids = torch.tensor(ids).view(SEQ_LENGTH).to(device)  # 转化为 tensor
    out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)  # 经过嵌入层得到隐藏状态
    position_ids = list(range(token_len)) + (SEQ_LENGTH - token_len) * [0]
    position_ids = torch.tensor([position_ids]).to(device)
    
    # 初始化 attention_mask
    attention_mask = torch.ones((SEQ_LENGTH, SEQ_LENGTH))
    for i in range(token_len):
        for j in range(token_len):
            if j <= i:
                attention_mask[i][j] = 0
    attention_mask = attention_mask.view(1, 1, SEQ_LENGTH, SEQ_LENGTH).to(device)
    
    # 存储每层的 key 和 value (这里只有第一层的缓存)
    k_cache = []
    v_cache = []

    # 只验证第一层
    out[:, token_len:] = 0  # 确保 padding 部分不参与推理
    out, k, v = block(out.to(dtype), position_ids, attention_mask)  # 第一层前向传播
    k_cache.append(k)  # 缓存 k
    v_cache.append(v)  # 缓存 v
    
    # 输出第一层的结果（用于验证模型的正确性）
    print(f"Layer 33 output: {out.shape}")
    # 在这里可以插入一些验证代码，如对比输出的值和预期结果

    # 从输出中截取最后一个 token
    out = out[:, token_len - 1:token_len].view(1, 1, HIDDEN_SIZE)  
    lm = LmHead()  # 语言模型头
    # 假设我们只关心第一个 token 位置
    # out = out[:, 0, :]  # 选取第一个 token 的输出，形状为 (1, 6144)
    # token = lm(out.to(dtype))  # 输入到 lm 层进行 token 预测
    # token = token.view(1)  # 获取 token 值

    # 假设我们只关心最后一个 token
    out = out[:, -1, :]  # 选取最后一个 token 的输出，形状为 (1, 6144)
    token = lm(out.to(dtype))  # 输入到 lm 层进行 token 预测
    token = token.view(-1)  # 获取 token 值
    print("token:", token)

    # token = torch.argmax(token, dim=-1)
    # print("token:", token)
    out_ids = [int(token[0])]  # 存储输出的 token id
    word = tokenizer._convert_id_to_token(int(token[0]))  # 转换为对应的词
    print(word, end="")

    # 验证生成过程：
    # while int(token[0]) != tokenizer.eos_token_id and token_len < ori_token_len + 10:
    #     token_len += 1
    #     input_ids = torch.tensor([token[0]]).to(device).long()  # 使用上一个 token 作为输入
    #     out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
    #     position_ids = torch.tensor([[token_len - 1]]).to(device)
        
    #     # 更新 attention_mask
    #     attention_mask = torch.zeros((1, 1, 1, SEQ_LENGTH + 1)).float().to(device)
    #     attention_mask[:, :, :, SEQ_LENGTH + 1 - token_len:] = 0
        
    #     # 只验证第一层
    #     out, k, v = block_kv(out.to(dtype), position_ids, attention_mask, k_cache[0].to(dtype), v_cache[0].to(dtype))
    #     k_cache[0][:SEQ_LENGTH - token_len] = 0  # 清空过期的缓存
    #     v_cache[0][:SEQ_LENGTH - token_len] = 0

    #     # 预测下一个 token
    #     # out = out[:, 0, :]  # 选取第一个 token 的输出，形状为 (1, 6144)
    #     # out = out.view(1, -1, HIDDEN_SIZE)  # 将 output reshape 成 (1, seq_len, hidden_size)
    #     token = lm(out.to(dtype)).view(1)
        

    #     out_ids.append(int(token))  # 存储生成的 token id
    #     word = tokenizer._convert_id_to_token(int(token[0]))  # 转换为对应的词
    #     print(word, end="")
    
    print("\noutput_ids:{}".format(out_ids))  # 输出生成的 token id 序列


# test_net_with_mask()
# test_first_layer_only()

# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)

# 查看 BlockCache 的 forward 方法的签名
print("BlockCache forward method signature:")
print(inspect.signature(BlockCache.forward))

print('Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
   convert_block(i)
   convert_block_cache(i)

print('Convert embedding')
convert_embedding()

print('Convert lm_head')
convert_lm_head()

print('convert_greedy_head')
convert_greedy_head()
print('convert_penalty_sample_head')
convert_penalty_sample_head()
