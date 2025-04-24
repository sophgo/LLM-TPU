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
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

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
        # SEQ_LENGTH = 175
        value_states = torch.zeros(
            (1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM), dtype=dtype).to(device)
        position_ids = torch.tensor(3*[[range(SEQ_LENGTH)]], dtype=torch.long).to(device)
        self.rotary_emb = self.layer.self_attn.rotary_emb
        self.cos, self.sin = self.rotary_emb(value_states, position_ids)
        self.cos = self.cos.transpose(1,2)
        self.sin = self.sin.transpose(1,2)

    def forward(self, hidden_states, position_ids, attention_mask):
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
        value_states = torch.randn(
            (1, 1, NUM_KEY_VALUE_HEADS, HEAD_DIM), dtype=dtype).to(device)
        position_ids = torch.tensor(3*[[range(SEQ_LENGTH)]], dtype=torch.long).to(device)
        self.rotary_emb = self.layer.self_attn.rotary_emb
        self.cos, self.sin = self.rotary_emb(value_states, position_ids)
        self.cos = self.cos.transpose(1,2)
        self.sin = self.sin.transpose(1,2)

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=(past_k, past_v),
            position_ids=position_ids,
            position_embeddings=(self.cos, self.sin),
            use_cache=True)
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()

class VisionTransformer(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pixel_values):
        patch_features = vision_backbone(pixel_values)
        projected_patch_embeddings = projector(patch_features)
        return projected_patch_embeddings


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
    

def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn(
        (1, SEQ_LENGTH, HIDDEN_SIZE)).float().to(device)
    position_ids = torch.tensor(
        3*[[range(SEQ_LENGTH)]], dtype=torch.long).to(device)
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
    position_ids = torch.tensor(3*[[range(1)]], dtype=torch.long).to(device)
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
    # make input
    x = torch.randn(1, 6, 224, 224).to(dtype=torch.float32, device=device)

    # export onnx
    model = VisionTransformer()
    torch.onnx.export(
        model, (x),
        f'{folder}/vit/vision_transformer.onnx',
        verbose=False,
        input_names=['pixel_values'],
        output_names=['hidden_states'],
        do_constant_folding=True,
        opset_version=15
    )

def convert_embedding():
    model = Embedding()
    input_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.int32).to(device)
    module = torch.jit.trace(model.forward, input_ids)
    torch.jit.save(module, f'{folder}/embedding.pt')


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


def setup_environment():
    import numpy as np
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return

def load_model():
    # setup environment
    setup_environment()

    # set
    device = torch.device(args.device)
    if args.device == "cpu":
        dtype = torch.float
        torch.set_num_threads(args.num_threads)
    else:
        # dtype = torch.float16
        dtype = torch.bfloat16

    # load model
    model_path = args.model_path

    origin_model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch.float,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cpu").eval()

    for param in origin_model.parameters():
        param.requires_grad = False
    return origin_model, device, dtype

def convert():
    # create folders to store onnx
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(folder + '/vit'):
        os.makedirs(folder + '/vit')

    # export models
    # print(f'Convert block & block_cache')
    # for i in tqdm(range(NUM_LAYERS)):
    #     convert_block(i)
    #     convert_block_cache(i)

    # print(f'Convert embedding')
    # convert_embedding()

    # print(f'Convert lm_head')
    # convert_lm_head()
    # convert_greedy_head()
    # convert_penalty_sample_head()

    print(f'Convert Vision Transformer')
    convert_vision_transformer()
    print("Done")

def test_model_generate(messages):
    # Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
    # > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from PIL import Image
    import torch

    # Load Processor & VLA
    model_path = "/workspace/models/openvla-7b"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        attn_implementation="eager",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.float,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cpu")

    breakpoint()
    # Grab image input & format prompt
    # image: Image.Image = get_from_camera(...)
    url = "./codalm3.png"
    image = Image.open(str(url)).convert("RGB")
    prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    inputs = processor(prompt, image).to("cpu", dtype=torch.float)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='export onnx')
    parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
    parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-n', '--num_threads', type=int, default=1, help='The number of threads used for torch if device is cpu')
    args = parser.parse_args()

    # processor & tokenizer
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    # load model
    origin_model, device, dtype = load_model()
    config = origin_model.config
    vision_backbone = origin_model.vision_backbone
    transformer = origin_model.language_model.model
    projector = origin_model.projector
    layers = transformer.layers
    llm_config = transformer.config

    SEQ_LENGTH = 256
    BATCH_SIZE = args.batch_size
    NUM_LAYERS = llm_config.num_hidden_layers
    HIDDEN_SIZE = llm_config.hidden_size
    NUM_ATTENTION_HEADS = llm_config.num_attention_heads
    NUM_KEY_VALUE_HEADS = llm_config.num_key_value_heads
    HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
    VOCAB_SIZE = llm_config.vocab_size
    print(f"Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n")
    print("\033[31m修改了load model方式，将attn_implementation由sdpa改为了eager，不然无法导出onnx\n\033[0m")
    folder = f"./tmp/onnx"

    # test_image(path = "./../python_demo/image1.jpg", resized_height=280, resized_width=420)
    # test_video(path = "./sample.mp4")

    # convert
    convert()