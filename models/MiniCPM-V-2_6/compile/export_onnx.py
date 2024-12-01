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
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

torch.set_grad_enabled(False)

PWD = os.getcwd().replace('\\', '/')
MiniCPMV_PATH = "{}/../../../../MiniCPM-V-2_6".format(PWD)

parser = argparse.ArgumentParser(description='export onnx')
parser.add_argument('-m', '--model_path', type=str,
                    default=MiniCPMV_PATH, help='path to the torch model')
parser.add_argument('-s', '--seq_length', type=int, default=512, help="sequence length")
parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
parser.add_argument('-n', '--num_threads', type=int, default=1, help='The number of threads used for torch if device is cpu')
parser.add_argument('-i', '--image_file', type=str, required=True, help='the size of the input for ViT will be based on the image file')

args = parser.parse_args()

model_path = args.model_path
folder = f"./tmp/onnx"

# set
device = torch.device(args.device)
if args.device == "cpu":
    dtype = torch.float
    torch.set_num_threads(args.num_threads)
else:
    dtype = torch.bfloat16

origin_model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, attn_implementation='eager',
    torch_dtype=dtype, device_map=args.device).eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

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
ID_EOS = config.eos_token_id
MAX_SLICE_NUMS = config.slice_config.max_slice_nums
image_file = args.image_file

print(f'Layers: {NUM_LAYERS}\nHidden size: {HIDDEN_SIZE}\n')


class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.embed = transformer.get_input_embeddings()

    def forward(self, input_ids):
        hidden_states = self.embed(input_ids)
        return hidden_states


class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        value_states = torch.randn(
            (1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)
        self.rotary_emb = self.layer.self_attn.rotary_emb
        self.cos, self.sin = self.rotary_emb(value_states, SEQ_LENGTH)
        self.cos = self.cos.view(SEQ_LENGTH, HEAD_DIM)
        self.sin = self.sin.view(SEQ_LENGTH, HEAD_DIM)

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids,
                                            use_cache=True,
                                            rotary_pos_emb_list=(
                                                self.cos, self.sin),
                                            )
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = layers[layer_id]
        value_states = torch.randn(
            (1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)
        self.rotary_emb = self.layer.self_attn.rotary_emb
        self.cos, self.sin = self.rotary_emb(value_states, SEQ_LENGTH)
        self.cos = self.cos.view(SEQ_LENGTH, HEAD_DIM)
        self.sin = self.sin.view(SEQ_LENGTH, HEAD_DIM)

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids=position_ids,
                                            past_key_value=(past_k, past_v),
                                            use_cache=True,
                                            rotary_pos_emb_list=(
                                                self.cos, self.sin),
                                            )
        present_k, present_v = past_kv
        return hidden_states.float(), present_k.float(), present_v.float()


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lm_head = origin_model.llm.get_output_embeddings()

    def forward(self, hidden_states):
        hidden_states = transformer.norm(hidden_states)
        m_logits = self.lm_head(hidden_states)
        _, token = torch.topk(m_logits.float(), 1)
        return token


class VisionTransformer(torch.nn.Module):
    def __init__(self, pixel_values, tgt_sizes):
        super().__init__()
        self.vpm = origin_model.vpm
        self.resampler = origin_model.resampler
        self.position_ids = self.vpm.embeddings.compute_position_ids(pixel_values, tgt_sizes)
        self.pos_embed = self.resampler.compute_pos_embed(pixel_values, tgt_sizes)

    def forward(self, pixel_values):
        hidden_states = self.vpm.embeddings(pixel_values, self.position_ids)
        encoder_outputs = self.vpm.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict="pt",
        )
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.vpm.post_layernorm(last_hidden_state)
        vit_embeds = self.resampler(last_hidden_state, self.pos_embed)
        return vit_embeds


def convert_vision_transformer():
    image = Image.open(image_file).convert('RGB')
    inputs = processor.image_processor([image], do_pad=True, max_slice_nums=MAX_SLICE_NUMS, return_tensors="pt")
    pixel_values = inputs["pixel_values"][0]
    for i in range(len(pixel_values)):
        pixel_values[i] = pixel_values[i].unsqueeze(0)
    pixel_values = torch.cat(pixel_values, dim=0).to(dtype).to(device)
    tgt_sizes = inputs["tgt_sizes"][0].to(dtype).to(device)

    model = VisionTransformer(pixel_values, tgt_sizes)

    torch.onnx.export(model, pixel_values,
                      f'{folder}/vision_transformer.onnx',
                      verbose=False,
                      input_names=['pixel_values'],
                      output_names=['vit_embeds'],
                      do_constant_folding=True,
                      opset_version=15)


def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE)).to(dtype).to(device)
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long).to(device)
    attention_mask = torch.ones(
        (1, 1, SEQ_LENGTH, SEQ_LENGTH)).to(dtype).to(device)
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
    hidden_states = torch.randn(1, HIDDEN_SIZE).to(dtype).to(device)

    module = torch.jit.trace(model.forward, hidden_states)
    torch.jit.save(module, f'{folder}/lm_head.pt')


# create folder to store onnx
if not os.path.exists(folder):
    os.makedirs(folder)


IMAGENET_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STD = (0.5, 0.5, 0.5)


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


def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    pixel_values = transform(image)
    return pixel_values


def test_net_with_mask():
    embed = Embedding()
    blocks = [Block(i) for i in range(NUM_LAYERS)]
    block_kvs = [BlockCache(i) for i in range(NUM_LAYERS)]

    image = Image.open(image_file).convert('RGB')
    inputs = processor.image_processor([image], do_pad=True, max_slice_nums=MAX_SLICE_NUMS, return_tensors="pt")
    pixel_values = inputs["pixel_values"][0]
    for i in range(len(pixel_values)):
        pixel_values[i] = pixel_values[i].unsqueeze(0)
    pixel_values = torch.cat(pixel_values, dim=0).to(dtype).to(device)
    tgt_sizes = inputs["tgt_sizes"][0].to(dtype).to(device)
    vit_infer = VisionTransformer(pixel_values, tgt_sizes)
    vit_embeds = vit_infer(pixel_values)  # [1, 64, 3584]
    vit_token_length = vit_embeds.shape[1]

    msgs = [{'role': 'user', 'content': '(<image>./</image>)\n请详细描述一下图片内容'}]
    prompts_lists = processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        prompts_lists, 
        [[image]], 
        max_slice_nums=MAX_SLICE_NUMS,
        use_image_id=None,
        return_tensors="pt", 
        max_length=8192
    ).to(device)
    ids = inputs.input_ids[0]
    first_offset = int(torch.where(ids==128244)[0][0])
    ids = ids.tolist()

    ID_IM_END = tokenizer.convert_tokens_to_ids("<|im_end|>")
    token_len = len(ids)
    ids = ids + (SEQ_LENGTH - token_len) * [0]
    input_ids = torch.tensor(ids).view(SEQ_LENGTH).to(device)
    out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)  # [1, 512, 3584]

    for i in range(vit_embeds.shape[0]):
        out[:, first_offset+i*vit_token_length:first_offset+(i+1)*vit_token_length, :] = vit_embeds[i]

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
    for i in range(NUM_LAYERS):
        out, k, v = blocks[i](out.to(dtype), position_ids,
                              attention_mask.to(dtype))
        k[:, :, token_len:, :] = 0
        v[:, :, token_len:, :] = 0
        k_cache.append(k)
        v_cache.append(v)

    out = out[:, token_len - 1:token_len].view(1, 1, HIDDEN_SIZE)
    lm = LmHead()
    token = lm(out.to(dtype)).view(1)
    out_ids = [int(token)]
    while int(token) not in [ID_EOS, ID_IM_END] and token_len < SEQ_LENGTH:
        token_len += 1
        input_ids = torch.tensor([token]).to(device)
        out = embed(input_ids).view(1, 1, HIDDEN_SIZE)
        position_ids = torch.tensor([[token_len - 1]]).to(device)
        attention_mask = torch.zeros(
            (1, 1, 1, SEQ_LENGTH + 1)).float().to(device)
        attention_mask[:, :, :, token_len-1:SEQ_LENGTH] = -10000.0
        for i in range(NUM_LAYERS):
            out, k, v = block_kvs[i](out.to(dtype), position_ids,
                                     attention_mask.to(dtype),
                                     k_cache[i].to(dtype), v_cache[i].to(dtype))
            k_cache[i][:, token_len-1:token_len, :, :] = k[:, :, :, :]
            v_cache[i][:, token_len-1:token_len, :, :] = v[:, :, :, :]
        token = lm(out.to(dtype)).view(1)
        out_ids.append(int(token))
    words = tokenizer.decode(out_ids)
    print(words)
    print("\noutput_ids:{}".format(out_ids))


# test_net_with_mask()
# exit()

# export models
print('Convert block & block_cache')
for i in tqdm(range(NUM_LAYERS)):
    convert_block_cache(i)
    convert_block(i)

print('Convert embedding')
convert_embedding()

print('Convert lm_head')
convert_lm_head()
print("Done!")

print('Convert Vision Transformer')
convert_vision_transformer()
print("Done!")
