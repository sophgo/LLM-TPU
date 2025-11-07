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
import torch.nn.functional as F
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, AutoTokenizer, DynamicCache
torch.set_grad_enabled(False)
import numpy as np
import onnxruntime

class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        out = model.embed_tokens(input_ids)
        return out.float()

class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        hidden_states = model.norm(hidden_states)
        m_logits = language_model.lm_head(hidden_states)
        return m_logits

class Block(torch.nn.Module):

    #q: torch.Size([1, 32, 599, 128])
    # cos: torch.Size([1, 1, 599, 128])

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        # SEQ_LENGTH = 175
        #value_states = torch.zeros(
        #    (1, 32, 599, 128), dtype=dtype).to(device)
        #position_ids = torch.ones((1, 599), dtype=torch.long).to(device)
        self.rotary_emb = self.layer.self_attn.rotary_emb.cpu()
        #self.cos, self.sin = self.rotary_emb(value_states, position_ids) # [1, 599, 128]
        #self.cos = self.cos.transpose(1,2)
        #self.sin = self.sin.transpose(1,2)

    def forward(self, hidden_states, position_ids, attention_mask):
        value_states = self.layer.self_attn.v_proj(hidden_states)
        #bsz, q_len, _ = hidden_states.size()
        self.cos, self.sin = self.rotary_emb(value_states, position_ids) # [1, 599, 128]


        hidden_states, past_kv = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=(self.cos, self.sin),
            use_cache=True)
        present_k, present_v = past_kv

        return hidden_states.float(), present_k.float(), present_v.float()


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        #value_states = torch.randn(
        #    (1, 32, 599, 128), dtype=dtype).to(device)
        #position_ids = torch.ones((1, 599), dtype=torch.long).to(device)
        self.rotary_emb = self.layer.self_attn.rotary_emb
        #self.cos, self.sin = self.rotary_emb(value_states, position_ids)
        #self.cos = self.cos.transpose(1,2)
        #self.sin = self.sin.transpose(1,2)

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        value_states = self.layer.self_attn.v_proj(hidden_states)
        #bsz, q_len, _ = hidden_states.size()
        self.cos, self.sin = self.rotary_emb(value_states, position_ids) # [1, 599, 128]
        hidden_states, past_kv = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=(past_k, past_v),
            position_ids=position_ids,
            position_embeddings=(self.cos, self.sin),
            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class GreedyHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, m_logits):
        _, token = torch.topk(m_logits.float(), 1)
        return token


def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn(
        (1, 599, HIDDEN_SIZE)).float().to(device)
    position_ids = torch.ones((1, 599), dtype=torch.long).to(device)
    attention_mask = torch.randn(
        (1, 599)).float().to(device)

    module = torch.jit.trace(model.forward, (hidden_states, position_ids, attention_mask))
    #torch.jit.save(module, f'{folder}/block_{layer_id}.pt')
    torch.onnx.export(
        model, (hidden_states, position_ids, attention_mask),
        f'{folder}/block_{layer_id}.onnx',
        verbose=False,
        input_names=['input_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states','past_k', 'past_v'],
        do_constant_folding=True,
        dynamic_axes={
            'position_ids': {0: 'batch', 1: 'seq'},
            'attention_mask': {0: 'batch', 1: 'seq'},
            'input_states': {0: 'batch', 1: 'seq'}
        },
        opset_version=15)


def convert_block_cache(layer_id):
    model = BlockCache(layer_id)
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).float().to(device)
    position_ids = torch.ones((1, 1), dtype=torch.long).to(device)
    attention_mask = torch.ones(
        (1, 1, 1, 600)).float().to(device)
    past_k = torch.randn((1, 32, 599, 128)).float().to(device)
    past_v = torch.randn((1, 32, 599, 128)).float().to(device)

    #module = torch.jit.trace(model.forward, (hidden_states, position_ids, attention_mask, past_k, past_v))
    #torch.jit.save(module, f'{folder}/block_cache_{layer_id}.pt')

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
        dynamic_axes={
            'position_ids': {0: 'batch', 1: 'seq'},
            'attention_mask': {0: 'batch', 3: 'seq'},
            'input_states': {0: 'batch', 2: 'seq'},
            'history_k': {0: 'batch', 2: 'seq'},
            'history_v': {0: 'batch', 2: 'seq'}
        },
        opset_version=15)


def convert_embedding():
    model = Embedding()
    input_ids = torch.ones([1, 599], dtype=torch.int32)
    #module = torch.jit.trace(model.forward, input_ids)
    #torch.jit.save(module, f'{folder}/embedding.pt')

    torch.onnx.export(
        model, (input_ids),
        f=f'{folder}/embedding.onnx',
        verbose=False,
        dynamo=True,
        external_data=True,
        input_names=['embed'],
        output_names=['feature'],
        opset_version=15,
        dynamic_axes={
            'embed': {0: 'batch', 1: 'seq'},
            'feature': {0: 'batch', 1: 'seq'}
        }
        )

def convert_cache_embedding():
    model = Embedding()
    #input_ids = torch.ones([1, 599], dtype=torch.int32)
    input_id = torch.tensor([599]).unsqueeze(0)
    #module = torch.jit.trace(model.forward, input_ids)
    #torch.jit.save(module, f'{folder}/embedding.pt')

    torch.onnx.export(
        model, (input_id),
        f=f'{folder}/embedding_cache.onnx',
        verbose=False,
        dynamo=True,
        external_data=True,
        input_names=['embed'],
        output_names=['feature'],
        opset_version=15,
        )


def convert_lm_head():
    model = LmHead()
    hidden_states = torch.randn(1, 1, HIDDEN_SIZE).float()
    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head.pt')
    """
    torch.onnx.export(
        model, (hidden_states),
        "lmhead_states.onnx",
        verbose=False,
        input_names=['lmhead'],
        output_names=['feature'],
        dynamo=True,
        external_data=True,
        opset_version=15,
        )
    """
def convert_greed():
    model = GreedyHead()
    #input_ids = torch.ones([1, 599], dtype=torch.int32)
    input_id = torch.tensor([115630]).unsqueeze(0).unsqueeze(0)
    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f'{folder}/greed.pt')



def setup_environment():
    import numpy as np
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return

def load_model():
    # setup environment
    setup_environment()

    # load model
    origin_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id, trust_remote_code=True, attn_implementation='eager',
        device_map="cpu"
    ).eval()

    for param in origin_model.parameters():
        param.requires_grad = False
    return origin_model, "cpu", None


# 构造 dummy 输入：假设输入为（batch_size, seq_len, feature_dim）
def export_audio_tower():
    onnx_dir = f'{folder}/audio'
    os.makedirs(onnx_dir, exist_ok=True)
    audio_tower = origin_model.audio_tower.cuda()

        # 2. 定义 ONNX 文件路径和输入输出名称
    onnx_path = os.path.join(onnx_dir, "audio_encoder.onnx")
    input_names = ["audio_features", "attention_mask"]
    output_names = ["audio_embeds"]

    # 3. 导出
    torch.onnx.export(
        audio_tower,
        (torch.randn([1, 128, 3000]).cuda(), torch.ones([1, 1, 1500, 1500]).cuda()),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        external_data=True,
        #do_constant_folding=True, # 子模块通常较小，可以开启
        opset_version=14,
    )
    print("export_audio_tower success ")
    import onnx
    onnx_model = onnx.load(onnx_path, load_external_data=True)
    os.remove("audio_ext_model.onnx")
    os.remove("audio_ext_model.onnx.data")
    final_onnx_path = "audio_ext_model.onnx"
    onnx.save_model(
        onnx_model,
        final_onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True, # <--- 这个参数在这里！
        location=f"{final_onnx_path}.data", # 指定合并后的外部数据文件名
        size_threshold=1024 # (可选) 只有大于1KB的张量才会被存为外部数据
    )

def export_multi_modal_projector():
    onnx_dir = f'{folder}/project'
    os.makedirs(onnx_dir, exist_ok=True)

    multi_modal_projector = origin_model.multi_modal_projector.cuda()
    multi_modal_projector.eval()
            # 2. 定义 ONNX 文件路径和输入输出名称
    onnx_path = os.path.join(onnx_dir, "multi_modal_projector.onnx")
    input_names = ["audio_multi_modal_projector_features"]
    output_names = ["audio_multi_modal_projector_embeds"]
    dummy_input_data = torch.from_numpy(np.random.randn(2, 750, 1280).astype(np.float32)).cuda()
    # 3. 导出
    torch.onnx.export(
        multi_modal_projector,
        dummy_input_data,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True, # 子模块通常较小，可以开启
        opset_version=14,
    )
    print("export_multi_modal_projector success ")


def convert():
    # export models
    print(f'Convert block & block_cache')
    #for i in tqdm(range(NUM_LAYERS)):
        #convert_block(i)
        #print("\033[31mexport success block\033[0m")
    #    convert_block_cache(i)
    #    print("\033[31mexport success block cache\033[0m")

    print(f'Convert embedding')

    #convert_cache_embedding()
    #convert_embedding()

    #print(f'Convert lm_head')
    #convert_lm_head()

    print(f'Convert audio')
    export_audio_tower()
    export_multi_modal_projector()
    print("Done")


def to_numpy(feat):
    return feat.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='export onnx')
    parser.add_argument('-m', '--model_path', type=str, default="Qwen/Qwen2-Audio-7B-Instruct", help='path to the torch model')
    parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument('-s', '--seq_length', type=int, default=599, help="sequence length")
    parser.add_argument('-vs', '--vision_seq_length', type=int, default=1024, help="vision input max sequence length")
    parser.add_argument('-n', '--num_threads', type=int, default=1, help='The number of threads used for torch if device is cpu')
    args = parser.parse_args()

    from modelscope import snapshot_download
    from transformers import AutoConfig
    model_id = snapshot_download("Qwen/Qwen2-Audio-7B-Instruct", cache_dir = '.', local_files_only=True)
    # processor & tokenizer
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # load model
    origin_model, device, dtype = load_model()
    config = origin_model.language_model.config
    language_model = origin_model.language_model
    audio = origin_model.audio_tower
    multi_modal_projector = origin_model.multi_modal_projector
    model = language_model.model
    layers = model.layers

    SEQ_LENGTH = args.seq_length
    VISION_SEQ_LENGTH = args.vision_seq_length
    BATCH_SIZE = 1 # args.batch_size
    NUM_LAYERS = config.num_hidden_layers
    HIDDEN_SIZE = config.hidden_size
    NUM_ATTENTION_HEADS = config.num_attention_heads
    NUM_KEY_VALUE_HEADS = config.num_key_value_heads
    HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
    VOCAB_SIZE = config.vocab_size
    config.attn_implementation="eager"
    print(config)
    print(f"Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n")
    print("\033[31m修改了load model方式，将attn_implementation由sdpa改为了eager，不然无法导出onnx\033[0m")

    # create folders to save onnx lor pt
    execution_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(execution_dir, "../convert_temp/onnx/") # folder for LLM
    if not os.path.exists(folder):
        os.makedirs(folder)
    convert()