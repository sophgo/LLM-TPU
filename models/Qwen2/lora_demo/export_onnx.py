#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import gc
import os
import json
import torch
import torch.nn as nn
import struct
import ctypes
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM
torch.set_grad_enabled(False)

class Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        out = transformer.embed_tokens(input_ids)
        return out.float()

class LoraEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_embedding_A = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=args.max_embedding_rank_num)
        self.lora_embedding_B = nn.Linear(in_features=args.max_embedding_rank_num, out_features=HIDDEN_SIZE, bias=False)

    def forward(self, input_ids, input_states):
        out = input_states + self.lora_embedding_B(self.lora_embedding_A(input_ids))
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
            use_cache=True,
            max_pos_len=args.max_pos_len
        )
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class BlockCache(torch.nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]

    def forward(self, hidden_states, position_ids, attention_mask, past_k, past_v):
        hidden_states, past_kv = self.layer(
            hidden_states,
            past_key_value=(past_k, past_v),
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            max_pos_len=args.max_pos_len
        )
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


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
    def __init__(self, top_k=50, min_tokens_to_keep=5):
        super().__init__()
        self.top_k = top_k
        self.min_tokens_to_keep = min_tokens_to_keep
        self.keep_matrix = torch.zeros((1, self.top_k), dtype=torch.bool)
        self.keep_matrix[0, : self.min_tokens_to_keep] = True

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
        filtered_logits = torch.where(mask, logits, torch.FloatTensor([-1000.0]))
        probs = filtered_logits.softmax(dim=1)
        return probs, token


def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((1, SHARE_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor([range(SHARE_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.randn((1, 1, SHARE_LENGTH, SHARE_LENGTH)).to(dtype).to(device)

    torch.onnx.export(
        model,
        (hidden_states, position_ids, attention_mask),
        f"{folder}/block_{layer_id}.onnx",
        verbose=False,
        input_names=["input_states", "position_ids", "attention_mask"],
        output_names=["hidden_states", "past_k", "past_v"],
        do_constant_folding=True,
        opset_version=15,
    )


def convert_block_cache(layer_id):
    model = BlockCache(layer_id)
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device)
    attention_mask = torch.ones((1, 1, 1, SEQ_LENGTH + 1)).to(dtype).to(device)
    past_k = (
        torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)
    )
    past_v = (
        torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)
    )

    torch.onnx.export(
        model,
        (hidden_states, position_ids, attention_mask, past_k, past_v),
        f"{folder}/block_cache_{layer_id}.onnx",
        verbose=False,
        input_names=[
            "input_states",
            "position_ids",
            "attention_mask",
            "history_k",
            "history_v",
        ],
        output_names=["hidden_states", "past_k", "past_v"],
        do_constant_folding=True,
        opset_version=15,
    )


def convert_embedding():
    model = Embedding()
    input_ids = torch.tensor([range(SHARE_LENGTH)], dtype=torch.int32).to(device)
    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f"{folder}/embedding.pt")


def convert_lm_head():
    model = LmHead()
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).to(dtype).to(device)
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f"{folder}/lm_head.pt")


def convert_greedy_head():
    model = GreedyHead()
    m_logits = torch.randn(1, VOCAB_SIZE)

    torch.onnx.export(
        model,
        (m_logits),
        f"{folder}/greedy_head.onnx",
        verbose=False,
        input_names=["m_logits"],
        output_names=["token"],
        do_constant_folding=True,
        opset_version=15,
    )


def convert_penalty_sample_head():
    model = PenaltySampleHead()
    m_logits = torch.randn(1, VOCAB_SIZE)
    input_ids = torch.tensor([range(SEQ_LENGTH)])
    top_p = torch.tensor([0.8])
    temperature = torch.tensor([0.98])
    penalty = torch.tensor([0.98])

    torch.onnx.export(
        model,
        (m_logits, input_ids, top_p, temperature, penalty),
        f"{folder}/penalty_sample_head.onnx",
        verbose=False,
        input_names=["m_logits", "input_ids", "top_p", "temperature", "penalty"],
        output_names=["probs", "token"],
        do_constant_folding=True,
        opset_version=15,
    )

def fp32_string(data):
    return bin(ctypes.c_uint32.from_buffer(ctypes.c_float(data)).value)[2:]

def convert_embedding_to_bit(path, transformer):
    print("\033[31m请注意！！如果embedding_mode=binary，目前convert_embedding_to_bit只支持embedding为float32格式，并且导出格式为bfloat16！！！\033[0m")
    print("\033[31m如果想导出float16的embedding，请修改此函数！！！\033[0m")
    embedding_weights = transformer.embed_tokens.weight.data
    embedding_weights_fp32 = embedding_weights.numpy().astype(np.float32).flatten()
    embedding_weights_uint32 = embedding_weights_fp32.view(np.uint32)
    embedding_weights_uint16 = (embedding_weights_uint32 >> 16).astype(np.uint16) # torch的格式必须是bfloat16才行
    if embedding_weights_uint16.dtype.byteorder == '>':
        embedding_weights_uint16 = embedding_weights_uint16.byteswap()
    embedding_weights_uint16 = embedding_weights_uint16.newbyteorder('little') # 确保数据以小端序存储
    embedding_weights_uint8 = embedding_weights_uint16.view(np.uint8)

    header = make_header(len(embedding_weights_uint8))

    embedding_weights_uint8 = np.concatenate([header, embedding_weights_uint8])
    with open(path, 'wb') as f:
        embedding_weights_uint8.tofile(f)

def file_md5(filename):
    import hashlib
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def check_md5_equality(file1, file2):
    md5_1 = file_md5(file1)
    md5_2 = file_md5(file2)
    if md5_1 != md5_2:
        raise ValueError(f"MD5 checksums do not match: {file1} does not match {file2}")
    else:
        print("MD5 checksumsm match successfully!")

def encrypt_and_save(data, save_path, args):
    import ctypes
    if not os.path.exists(args.lib_path):
        raise FileNotFoundError(f"{args.lib_path} not found")
    lib = ctypes.CDLL(args.lib_path)
    lib.encrypt.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    lib.encrypt.restype = ctypes.POINTER(ctypes.c_uint8)

    input_data = data.astype(np.uint8)
    input_bytes = input_data.nbytes
    output_bytes = ctypes.c_uint64()

    input_data_ctypes = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    encrypted_data_ptr = lib.encrypt(input_data_ctypes, input_bytes, ctypes.byref(output_bytes))
    encrypted_data = np.ctypeslib.as_array(encrypted_data_ptr, shape=(output_bytes.value,))

    with open(save_path, 'wb') as f:
        f.write(encrypted_data)

    lib.free_memory(encrypted_data_ptr)

def decrypt_and_save(data, save_path, args):
    if not os.path.exists(args.lib_path):
        raise FileNotFoundError(f"{args.lib_path} not found")
    lib = ctypes.CDLL(args.lib_path)

    lib.decrypt.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    lib.decrypt.restype = ctypes.POINTER(ctypes.c_uint8)
    lib.free_memory.argtypes = [ctypes.POINTER(ctypes.c_uint8)]

    input_data = data.astype(np.uint8)
    input_bytes = input_data.nbytes
    output_bytes = ctypes.c_uint64()

    input_data_ctypes = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    decrypted_data_ptr = lib.decrypt(input_data_ctypes, input_bytes, ctypes.byref(output_bytes))
    decrypted_data = np.ctypeslib.as_array(decrypted_data_ptr, shape=(output_bytes.value,))

    with open(save_path, 'wb') as f:
        f.write(decrypted_data.tobytes())

    lib.free_memory(decrypted_data_ptr)


def load_lora_model(origin_model, path):
    import copy
    from peft import LoraConfig, PeftModel
    # 1. load lora
    config_file = os.path.join(path, "adapter_config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Neither config.json nor adapter_config.json found in {path}")
    with open(config_file) as f:
        lora_config_dict = json.load(f)
    lora_config = LoraConfig(**lora_config_dict)
    lora_model = PeftModel.from_pretrained(copy.deepcopy(origin_model), path, offload_dir='./offload_dir') # 需要做deepcopy，不然会影响origin_model
    return lora_model, lora_config


def convert_lora_to_bit(lora_model, lora_config, args):
    # extract layer from model
    lora_weight_list = []
    for i in range(len(lora_model.base_model.model.model.layers)):
        lora_layers = lora_model.base_model.model.model.layers[i]
        extracted_layers = {}

        for name, module in lora_layers.named_modules():
            if 'lora_A.default' in name or 'lora_B.default' in name:
                if any(layer_name in name for layer_name in list(lora_config.target_modules)):
                    extracted_layers[name] = module

        lora_A_weight_list = []
        lora_B_weight_list = []

        for name, extracted_layer in extracted_layers.items():
            lora_weight = extracted_layer.weight.detach().cpu().numpy().transpose(1,0)
            left_dim, right_dim = lora_weight.shape

            if 'lora_A' in name and left_dim > right_dim:
                new_lora_weight = np.zeros((left_dim, args.max_rank_num), dtype=np.float32)
                new_lora_weight[:, :right_dim] = lora_weight
                lora_A_weight_list.append(new_lora_weight)
            elif 'lora_B' in name and left_dim < right_dim:
                new_lora_weight = np.zeros((args.max_rank_num, right_dim), dtype=np.float32)
                new_lora_weight[:left_dim, :] = lora_weight
                lora_B_weight_list.append(new_lora_weight)
            else:
                raise NotImplementedError

        # 由于在final.mlir中，weight的权重排列顺序是[lora_B, lora_A, lora_B, lora_A]的形式
        # 所以需要把B排列在前面
        for a, b in zip(lora_A_weight_list, lora_B_weight_list):
            lora_weight_list.append(a)
            lora_weight_list.append(b)

    # Flatten the weights and convert to uint32
    lora_weights_fp32 = np.concatenate([w.flatten() for w in lora_weight_list])
    lora_weights_fp32 = lora_weights_fp32
    lora_weights_uint32 = lora_weights_fp32.view(np.uint32)
    lora_weights_uint16 = (lora_weights_uint32 >> 16).astype(np.uint16)  # Convert to bfloat16

    if lora_weights_uint16.dtype.byteorder == '>':
        lora_weights_uint16 = lora_weights_uint16.byteswap()
    lora_weights_uint16 = lora_weights_uint16.newbyteorder('little')  # Ensure little-endian storage

    lora_weights_uint8_low = (lora_weights_uint16 >> 8).astype(np.uint8)
    lora_weights_uint8_high = (lora_weights_uint16 & 0xFF).astype(np.uint8)
    lora_weights_uint8 = np.column_stack((lora_weights_uint8_high, lora_weights_uint8_low)).reshape(-1)

    return lora_weights_uint8

def convert_lora_embedding():
    model = LoraEmbedding()
    input_ids = torch.tensor([range(SEQ_LENGTH)])
    input_states = torch.randn(1, SEQ_LENGTH, HIDDEN_SIZE)

    torch.onnx.export(
        model,
        (input_ids, input_states),
        f"{folder}/lora_embedding.onnx",
        verbose=False,
        input_names=["input_ids", "input_states"],
        output_names=["hidden_states"],
        do_constant_folding=True,
        opset_version=15,
    )


def convert_lora_embedding_to_bit(lora_model, lora_config, args):
    # extract layer from model
    lora_weight_list = []
    lora_layers = lora_model.base_model.model.model.embed_tokens
    extracted_layers = {}
    for name, module in lora_layers.named_modules():
        if 'lora_embedding_A' in name or 'lora_embedding_B' in name:
            extracted_layers[name] = module.default

    lora_A_weight_list = []
    lora_B_weight_list = []

    for name, extracted_layer in extracted_layers.items():
        lora_weight = extracted_layer.detach().cpu().numpy().transpose(1,0)
        left_dim, right_dim = lora_weight.shape

        if 'lora_embedding_A' in name and left_dim > right_dim:
            new_lora_weight = np.zeros((left_dim, args.max_embedding_rank_num), dtype=np.float32)
            new_lora_weight[:, :right_dim] = lora_weight
            lora_A_weight_list.append(new_lora_weight)
        elif 'lora_embedding_B' in name and left_dim < right_dim:
            new_lora_weight = np.zeros((args.max_embedding_rank_num, right_dim), dtype=np.float32)
            new_lora_weight[:left_dim, :] = lora_weight
            lora_B_weight_list.append(new_lora_weight)
        else:
            raise NotImplementedError

    # 由于在final.mlir中，weight的权重排列顺序是[lora_B, lora_A]的形式
    # 但是在加载时，是按照算子调用逻辑来调用的，lora_A先走先调，lora_B后跑后调
    # 所以需要把A排列在前面
    for a, b in zip(lora_A_weight_list, lora_B_weight_list):
        lora_weight_list.append(a)
        lora_weight_list.append(b)

    # Flatten the weights and convert to uint32
    lora_weights_fp32 = np.concatenate([w.flatten() for w in lora_weight_list])
    lora_weights_uint32 = lora_weights_fp32.view(np.uint32)
    lora_weights_uint16 = (lora_weights_uint32 >> 16).astype(np.uint16)  # Convert to bfloat16

    if lora_weights_uint16.dtype.byteorder == '>':
        lora_weights_uint16 = lora_weights_uint16.byteswap()
    lora_weights_uint16 = lora_weights_uint16.newbyteorder('little')  # Ensure little-endian storage

    lora_weights_uint8_low = (lora_weights_uint16 >> 8).astype(np.uint8)
    lora_weights_uint8_high = (lora_weights_uint16 & 0xFF).astype(np.uint8)
    lora_weights_uint8 = np.column_stack((lora_weights_uint8_high, lora_weights_uint8_low)).reshape(-1)

    return lora_weights_uint8

def make_header(size, header_size = 64):
    if header_size < 8:
        raise ValueError("Header size must be at least 4 bytes to store the size.")
    header = np.zeros(header_size, dtype=np.uint8)
    size_bytes = struct.pack('<Q', header_size + size)
    header[:8] = np.frombuffer(size_bytes, dtype=np.uint8)
    return header

def convert_total_lora_to_bit(encrypt_path, origin_model, args):
    if args.max_rank_num == 0:
        raise ValueError(f"max_rank_num is equal to {args.max_rank_num}")
    if args.max_embedding_rank_num == 0:
        raise ValueError(f"max_embedding_rank_num is equal to {args.max_embedding_rank_num}")

    # path
    origin_path = "lora_weights.bin"
    decrypt_path = "decrypted_lora_weights.bin"

    # lora embedding
    lora_model, lora_config = load_lora_model(origin_model, args.lora_embedding_path)
    lora_embedding_weights = convert_lora_embedding_to_bit(lora_model, lora_config, args)
    # lora
    lora_model, lora_config = load_lora_model(origin_model, args.lora_path)
    lora_weights = convert_lora_to_bit(lora_model, lora_config, args)
    # header
    header = make_header(len(lora_weights) + len(lora_embedding_weights))
    total_lora_weights = np.concatenate([header, lora_weights, lora_embedding_weights]) # 由于在bmodel中，lora_embedding放在后面，因此这里是lora,lora_embedding的顺序

    # save and encrypt & decrypt
    with open(origin_path, 'wb') as f:
        total_lora_weights.tofile(f)
    # encrypt
    encrypt_and_save(total_lora_weights, encrypt_path, args)
    # decrypt
    encrypted_data = np.fromfile(encrypt_path, dtype=np.uint8)
    decrypt_and_save(encrypted_data, decrypt_path, args)
    check_md5_equality(origin_path, decrypt_path)


def setup_environment():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return

def load_model(args):
    # setup environment
    setup_environment()

    # set
    device = torch.device(args.device)
    if args.device == "cpu":
        dtype = torch.float
        torch.set_num_threads(args.num_threads)
    else:
        dtype = torch.bfloat16

    # load model
    model_path = args.model_path
    origin_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype
    ).eval().to(device)
    for param in origin_model.parameters():
        param.requires_grad = False
    return origin_model, device, dtype

def convert():
    # create folder to store onnx
    if not os.path.exists(folder):
        os.makedirs(folder)

    # export lora model
    print("Convert lora")
    convert_total_lora_to_bit("encrypted_lora_weights.bin", origin_model, args)

    print("Convert lora embedding")
    convert_lora_embedding()

    # export models
    print("Convert block & block_cache")
    for i in tqdm(range(NUM_LAYERS)):
        convert_block(i)
        convert_block_cache(i)

    print('Convert embedding')
    if args.embedding_mode == "default":
        convert_embedding()
    elif args.embedding_mode == "binary":
        convert_embedding_to_bit('embedding.bin', transformer)

    print("Convert lm_head")
    convert_lm_head()
    convert_greedy_head()
    convert_penalty_sample_head()


    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='export onnx')
    parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
    parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-s', '--seq_length', type=int, default=512, help="sequence length")
    parser.add_argument('-n', '--num_threads', type=int, default=1, help='The number of threads used for torch if device is cpu')
    parser.add_argument('--prefill_length', type=int, default=6144, help="prefill length")
    parser.add_argument('--max_pos_len', type=int, default=8704, help="max position length")
    parser.add_argument('--generation_mode', type=str, default="default", choices=["default", "lmhead_with_penalty", "lmhead_with_sample", "lmhead_with_top1"], help="generation mode")
    parser.add_argument('--embedding_mode', type=str, default="default", choices=["default", "binary"], help="if set embedding_mode=binary, will save embedding.bin and infer without tpu")
    parser.add_argument('--lib_path', type=str, default='', help='lib path by user')
    parser.add_argument('--lora_path', type=str, default="", help="path to the lora model")
    parser.add_argument('--lora_embedding_path', type=str, default="", help="path to the lora embedding model")
    parser.add_argument('--max_rank_num', type=int, default=0, help="the max rank for lora model")
    parser.add_argument('--max_embedding_rank_num', type=int, default=0, help="the max rank for lora embedding model")
    args = parser.parse_args()

    # load model
    origin_model, device, dtype = load_model(args)
    config = origin_model.config
    transformer = origin_model.model
    layers = transformer.layers
    SEQ_LENGTH = args.seq_length
    SHARE_LENGTH = args.prefill_length
    BATCH_SIZE = args.batch_size
    NUM_LAYERS = config.num_hidden_layers
    HIDDEN_SIZE = config.hidden_size
    NUM_ATTENTION_HEADS = config.num_attention_heads
    NUM_KEY_VALUE_HEADS = config.num_key_value_heads
    HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
    VOCAB_SIZE = config.vocab_size
    print(f"Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n")
    folder = f"./tmp_prefill{args.prefill_length}_seq{args.seq_length}/onnx"

    # convert
    convert()
