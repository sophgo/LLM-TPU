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
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

torch.set_grad_enabled(False)

# PWD = os.getcwd().replace('\\', '/')
# Molmo_PATH = "{}/../../../../Molmo-7B-D-0924".format(PWD)
Molmo_PATH = '/workspace/Molmo-7B/Molmo-7B-D-0924'

parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('--model_path', type=str,
                    default=Molmo_PATH, help='path to the torch model')
parser.add_argument('--image_size', type=int,
                    default=334, help='image_size')
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")

args = parser.parse_args()

device = torch.device(args.device)
if args.device == "cpu":
    dtype = torch.float
else:
    dtype = torch.bfloat16

model_path = args.model_path
folder = f"./tmp/onnx"

processor = AutoProcessor.from_pretrained(
    '/workspace/Molmo-7B/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

origin_model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, attn_implementation='eager',
    torch_dtype=dtype).eval()

tokenizer=processor.tokenizer

for param in origin_model.parameters():
    param.requires_grad = False

config = origin_model.config
transformer = origin_model.model.transformer
layers = transformer.blocks

SEQ_LENGTH = config.max_position_embeddings
NUM_LAYERS = config.num_hidden_layers
HIDDEN_SIZE = config.hidden_size
NUM_ATTENTION_HEADS = config.num_attention_heads
NUM_KEY_VALUE_HEADS = config.num_key_value_heads
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
use_cache = config.use_cache
IMAGE_SIZE = args.image_size
device = origin_model.device

print(f'Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n')

class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        hidden_states = transformer.wte(input_ids)
        return hidden_states

class EmbDrop(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        emb_drop = transformer.emb_drop(input_ids)
        return emb_drop

class RMSLayerNorm(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        rmsLayerNorm = transformer.ln_f(input_ids)
        return emb_drop

class RMSLayerNorm(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        rmsLayerNorm = transformer.ln_f(input_ids)
        return emb_drop

class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = self.layer.rotary_emb
        self.sin,self.cos = self.rotary_emb.get_rotary_embedding(SEQ_LENGTH,device)

    def forward(self, hidden_states, position_ids, attention_bias):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_bias,
                                            position_ids,
                                            use_cache=True,
                                            pos_sin=self.sin,
                                            pos_cos=self.cos
                                            )
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        self.rotary_emb = self.layer.rotary_emb
        self.sin,self.cos = self.rotary_emb.get_rotary_embedding(SEQ_LENGTH,device)

    def forward(self, hidden_states, position_ids, attention_bias, past_k,
                past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_bias,
                                            position_ids,
                                            use_cache=True,
                                            pos_sin=self.sin,
                                            pos_cos=self.cos,
                                            layer_past=(past_k, past_v),
                                            )
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        m_logits = transformer.ff_out(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token


class VisionTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = origin_model.model.vision_backbone

    def forward(self,image,image_mask):
        image_features, cls_emb= self.vision_backbone(image,image_mask)
        return image_features, cls_emb


def convert_vision_transformer():
    model = VisionBackbone()
    image = torch.randn(
        (IMAGE_SIZE, IMAGE_SIZE,3))
    torch.onnx.export(model, image,
                      f'{folder}/vision_transformer.onnx',
                      verbose=False,
                      input_names=['image'],
                      output_names=['image_features'],
                      do_constant_folding=True,
                      opset_version=15)


def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE)).bfloat16()
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long)
    attention_mask = torch.ones(
        (1, 1, SEQ_LENGTH, SEQ_LENGTH)).bfloat16()
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
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE)).bfloat16()
    position_ids = torch.tensor([range(1)], dtype=torch.long)
    attention_mask = torch.ones((1, 1, 1, SEQ_LENGTH + 1)).bfloat16()
    past_k = torch.randn(
        (1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).bfloat16()
    past_v = torch.randn(
        (1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).bfloat16()

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
    input_ids = torch.tensor([range(SEQ_LENGTH)])

    torch.onnx.export(model, (input_ids),
                      f'{folder}/embedding.onnx',
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['input_embed'],
                      do_constant_folding=True,
                      opset_version=15)


def convert_lm_head():
    model = LmHead()
    input = torch.randn(1, HIDDEN_SIZE).bfloat16()

    torch.onnx.export(model, (input),
                      f'{folder}/lm_head.onnx',
                      verbose=False,
                      input_names=['hidden_states'],
                      output_names=['m_logits'],
                      do_constant_folding=True,
                      opset_version=15)


# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)


IMAGENET_MEAN = (0.48145466,0.4578275,0.40821073)
IMAGENET_STD = (0.26862954,0.26130258,0.27577711)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def load_image(image_file, input_size=336):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    pixel_values = transform(image)
    return pixel_values


def test_net_with_mask():
    embed = Embedding()
    blocks = [Block(i) for i in range(NUM_LAYERS)]
    block_kvs = [BlockCache(i) for i in range(NUM_LAYERS)]
    vit_infer = VisionTransformer()
    RMS = RMSLayerNorm()
    embdrop = EmbDrop()
    image_path = "/workspace/dog.jpg"
    image = Image.open(image_path)

    new_size = (1000, 1000)
    resized_image = image.resize(new_size, Image.LANCZOS) 

    inputs = processor.process(
        images=[resized_image],
        text="Describe this image."
    )
    inputs = {k: v.to(origin_model.device).unsqueeze(0) for k, v in inputs.items()}
    input_ids = inputs['input_ids']
    image_input_idx = inputs['image_input_idx']
    batch_size, seq_len = input_ids.size()
    max_new_tokens = 200
    stop_strings="<|endoftext|>"
    mask_len = seq_len + max_new_tokens
    position_ids: Optional[torch.Tensor] = None
    append_last_valid_logits: Optional[torch.Tensor] = None
    prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>"
    prefix_ids = tokenizer.encode(prefix)
    query = "</image>\nWhat is in the image?<|im_end|>\n<|im_start|>assistant\n"
    query_ids = tokenizer.encode(query)
    image_ids = [0] * 64
    prefix_len = len(prefix_ids)
    ids = prefix_ids + image_ids + query_ids
    token_len = len(ids)
    output_hidden_states = False
    batch_size, seq_len = input_ids.size()
    past_length = 0

    input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
    x = embed(input_ids)

    num_image: Optional[int] = None
    images = inputs['images']
    image_masks = inputs['image_masks']

    if images is not None:
        image_features, cls_embed = vit_infer(images, image_masks)
        num_image, num_patch = image_features.shape[1:3]
        assert image_input_idx.shape == (batch_size, num_image, num_patch)

        # inster the image feature into the embedding.
        image_features = image_features.view(batch_size, num_image * num_patch, -1)
        image_input_idx = image_input_idx.view(batch_size, num_image * num_patch)

        valid = image_input_idx >= 0
        batch_idx = torch.arange(batch_size, device=x.device)
        batch_idx = torch.tile(batch_idx[:, None], [1, image_features.shape[1]])

        # For hf demo/endpoint
        image_features = image_features.to(x.device)

        x[batch_idx[valid], image_input_idx[valid]] += image_features[valid]

    x = embdrop(x)  # type: ignore

    attention_mask = torch.ones((SEQ_LENGTH, SEQ_LENGTH)).float() * -10000.0
    for i in range(token_len):
        for j in range(token_len):
            if j <= i:
                attention_mask[i][j] = 0.0
    attention_mask = attention_mask.view(
        1, 1, SEQ_LENGTH, SEQ_LENGTH)
    k_cache = []
    v_cache = []

    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

    # decoder layers
    all_hidden_states = []
        
    # Apply blocks one-by-one.
    for i in range(NUM_LAYERS):
        if output_hidden_states:
            # add hidden states
            all_hidden_states.append(x)
        layer_past = None
        x, k, v = blocks[i](x, attention_bias=attention_mask, position_ids=position_ids)
        k[:, :, token_len:, :] = 0
        v[:, :, token_len:, :] = 0
        k_cache.append(k)
        v_cache.append(v)

        if attn_key_values is not None:
            assert cache is not None
            attn_key_values.append(cache)

    # if last_logits_only:
        # shape: (batch_size, 1, d_model)
        if append_last_valid_logits is not None:
            last_valid_output = x[
                torch.arange(x.shape[0], device=x.device), append_last_valid_logits.to(x.device)]
            x = last_valid_output.unsqueeze(1)
        else:
            x = x[:, -1, :].unsqueeze(1)

    x = RMS(x)
    if output_hidden_states:
        # add final hidden state post-final-layernorm, following HuggingFace's convention
        all_hidden_states.append(x)

    # Get logits.
    # shape: (batch_size, seq_len or 1, vocab_size)
    lm = LmHead()
    token = lm(x).view(1)
    out_ids = [int(token)]
    
    while token_len < SEQ_LENGTH:
        token_len += 1
        input_ids = torch.tensor([token])
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]])
        attention_mask = torch.zeros(
            (1, 1, 1, SEQ_LENGTH + 1)).float()
        attention_mask[:, :, :, token_len-1:SEQ_LENGTH] = -10000.0
        for i in range(NUM_LAYERS):
            out, k, v = block_kvs[i](out.bfloat16(), attention_bias,
                                     position_ids.bfloat16(),
                                     k_cache[i].bfloat16(), v_cache[i].bfloat16())
            k_cache[i][:, token_len-1:token_len, :, :] = k[:, :, :, :]
            v_cache[i][:, token_len-1:token_len, :, :] = v[:, :, :, :]
        token = lm(out.bfloat16()).view(1)
        out_ids.append(int(token))
    words = tokenizer.decode(out_ids)
    print(words)
    print("\noutput_ids:{}".format(out_ids))

test_net_with_mask()
# exit()

# export models
# print(f'Convert block & block_cache')
# for i in tqdm(range(NUM_LAYERS)):
#     convert_block_cache(i)
#     convert_block(i)

# print(f'Convert embedding')
# convert_embedding()

# print(f'Convert lm_head')
# convert_lm_head()
# print("Done!")

# print(f'Convert Vision Transformer')
# convert_vision_transformer()
# print("Done!")
