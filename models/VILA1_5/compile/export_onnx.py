#!/usr/bin/env python3
# ==============================================================================
#
# Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================


from builder import load_pretrained_model
import torch
import os
import argparse
import numpy as np

from transformers import SiglipImageProcessor, AutoTokenizer
from transformers import AutoConfig, AutoModelForCausalLM, SiglipVisionModel
from llava.model.multimodal_projector.base_projector import MultimodalProjector, MultimodalProjectorConfig
torch.set_grad_enabled(False)

class Vision_Embedding(torch.nn.Module):
    def __init__(self, vision_tower, mm_projector):
        super().__init__()
        self.vision_tower = vision_tower
        self.mm_projector = mm_projector

    def forward(self, images):
        image_forward_out = self.vision_tower(
            images.to(device),
            output_hidden_states=True,
        )
        image_feature = image_forward_out.hidden_states[-2]
        return self.mm_projector(image_feature)


class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        return llm_model.embed_tokens(input_ids)


class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = llama.layers[layer_id]

        self.rotary_emb = self.layer.self_attn.rotary_emb
        value_states = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)
        position_ids = torch.tensor([range(SEQ_LENGTH)],dtype=torch.long).to(device)
        self.cos, self.sin = self.rotary_emb(value_states, position_ids)
        self.cos = self.cos.view(SEQ_LENGTH, HEAD_DIM)
        self.sin = self.sin.view(SEQ_LENGTH, HEAD_DIM)

    def forward(self, hidden_states, position_ids, attention_mask):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids,
                                            use_cache=True,
                                            position_embeddings=(self.cos, self.sin))
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v
    
class BlockCache(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = llama.layers[layer_id]

        self.rotary_emb = self.layer.self_attn.rotary_emb
        value_states = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM)).to(dtype).to(device)
        position_ids = torch.tensor([range(SEQ_LENGTH)],dtype=torch.long).to(device)
        self.cos, self.sin = self.rotary_emb(value_states, position_ids)
        self.cos = self.cos.view(SEQ_LENGTH, HEAD_DIM)
        self.sin = self.sin.view(SEQ_LENGTH, HEAD_DIM)

    def forward(self, hidden_states, position_ids, attention_mask, past_k,
                past_v):
        hidden_states, past_kv = self.layer(hidden_states,
                                            attention_mask,
                                            position_ids=position_ids,
                                            past_key_value=(past_k, past_v),
                                            use_cache=True,
                                            position_embeddings=(self.cos, self.sin))
        present_k, present_v = past_kv
        return hidden_states, present_k, present_v

class LmHead(torch.nn.Module):

    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def forward(self, hidden_states):
        hidden_states = self.llm.model.norm(hidden_states)
        m_logits = self.llm.lm_head(hidden_states)
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
        
def convert_vision_embedding():
    model = Vision_Embedding(vision_tower.vision_model, mm_projector)
    images = torch.randn(NUM_FRAMES, 3, 384, 384, dtype=dtype).to(device=device)
    torch.onnx.export(
        model, images,
        f"{folder}/vision_embedding.onnx",
        input_names=["images"],
        output_names=["hidden_states"],
        do_constant_folding=True,
        opset_version=15
    )

def convert_embedding():
    model = Embedding()
    input_ids = torch.tensor([range(SEQ_LENGTH)]).to(device=device)

    torch.onnx.export(model, (input_ids),
                      f'{folder}/embedding.onnx',
                      verbose=False,
                      input_names=['input_ids'],
                      output_names=['input_embed'],
                      do_constant_folding=True,
                      opset_version=15)


def convert_block(layer_id):
    model = Block(layer_id)
    hidden_states = torch.randn((1, SEQ_LENGTH, HIDDEN_SIZE), dtype=dtype).to(device=device)
    position_ids = torch.tensor([range(SEQ_LENGTH)], dtype=torch.long).to(device=device)
    attention_mask = -1000 * torch.ones((1, 1, SEQ_LENGTH, SEQ_LENGTH), dtype=torch.float32).triu(diagonal=1).to(device=device)

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
    hidden_states = torch.randn((1, 1, HIDDEN_SIZE), dtype=dtype).to(device=device)
    position_ids = torch.tensor([range(1)], dtype=torch.long).to(device=device)
    attention_mask = -1000 * torch.ones((1, 1, 1, SEQ_LENGTH + 1), dtype=torch.float32).triu(diagonal=1).to(device=device)
    past_k = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM), dtype=dtype).to(device=device)
    past_v = torch.randn((1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HEAD_DIM), dtype=dtype).to(device=device)

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

def convert_lm_head():
    model = LmHead(llm_model)
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

def opencv_extract_frames(video_file, num_frames):
    import cv2
    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    frame_interval = frame_count // num_frames
    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    images = []
    count = 0
    success = True
    while success:
        # print("frame_count:", frame_count, "count:", count, "num_frames:", num_frames, "frame_interval:", frame_interval)
        if frame_count >= num_frames:
            # breakpoint()
            # vidcap.set(cv2.CAP_PROP_POS_FRAMES, 200)
            success, frame = vidcap.read()
            if count in frame_indices:
                try:
                    # breakpoint()
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    images.append(im_pil)
                except BaseException:
                    continue
                if len(images) >= num_frames:
                    return images, num_frames
            count += 1
        else:
            # Left padding frames if the video is not long enough
            success, frame = vidcap.read()
            if success:
                try:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    images.append(im_pil)
                except BaseException:
                    continue
                count += 1
            else:
                break

def test_net_with_mask():
    video_path = "sample.mp4"

    embed = Embedding()
    blocks = [Block(i) for i in range(NUM_LAYERS)]
    block_kvs = [BlockCache(i) for i in range(NUM_LAYERS)]

    image_processor = SiglipImageProcessor.from_pretrained(vision_path)
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    breakpoint()
    images, _ = opencv_extract_frames(video_path, 10)
    breakpoint()

    image = Image.open(image_file).convert('RGB')
    inputs = processor.image_processor([image], do_pad=True, max_slice_nums=MAX_SLICE_NUMS, return_tensors="pt")
    pixel_values = inputs["pixel_values"][0]
    for i in range(len(pixel_values)):
        pixel_values[i] = pixel_values[i].unsqueeze(0)
    pixel_values = torch.cat(pixel_values, dim=0).to(dtype).to(device)
    tgt_sizes = inputs["tgt_sizes"][0].to(dtype).to(device)
    vit_infer = VisionTransformer(pixel_values, tgt_sizes)
    vit_embeds = vit_infer(pixel_values)  # [1, 64, 3584]

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
    image_offsets = torch.where(ids==128244)[0].tolist()
    ids = ids.tolist()

    ID_IM_END = tokenizer.convert_tokens_to_ids("<|im_end|>")
    token_len = len(ids)
    ids = ids + (SEQ_LENGTH - token_len) * [0]
    input_ids = torch.tensor(ids).view(SEQ_LENGTH).to(device)
    out = embed(input_ids).view(1, SEQ_LENGTH, HIDDEN_SIZE)  # [1, 512, 3584]

    patch_num = pixel_values.shape[0]
    patch_size = len(image_offsets) // patch_num
    for i in range(patch_num):
        out[:, image_offsets[i*patch_size]:image_offsets[i*patch_size]+patch_size, :] = vit_embeds[i]

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

def convert():
    # create folder to store onnx
    if not os.path.exists(folder):
        os.makedirs(folder)

    # convert_vision_embedding()
    # convert_embedding()
    # convert_lm_head()
    # convert_greedy_head()
    # convert_penalty_sample_head()

    for i in range(16,17):
        convert_block(i)
        convert_block_cache(i)
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='export onnx')
    parser.add_argument('-m', '--model_path', type=str, help='path to the torch model')
    parser.add_argument('-s', '--seq_length', type=int, default=4096, help="sequence length")
    parser.add_argument('-d', '--device', type=str, choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    model_path = args.model_path
    device = torch.device(args.device)
    if args.device == "cpu":
        dtype = torch.float
    else:
        dtype = llm_cfg.torch_dtype

    model_name = "vila1.5-3b"
    llm_path = os.path.join(model_path, "llm")
    vision_path = os.path.join(model_path, "vision_tower")
    proj_path = os.path.join(model_path, "mm_projector")
    config = AutoConfig.from_pretrained(model_path)
    llm_cfg = AutoConfig.from_pretrained(llm_path)

    if args.device == "cpu":
        vision_tower = SiglipVisionModel.from_pretrained(vision_path, torch_dtype=dtype).eval().to(device)
        mm_projector = MultimodalProjector.from_pretrained(proj_path, config, torch_dtype=dtype).eval().to(device)
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_path, config=llm_cfg, torch_dtype=dtype, attn_implementation='eager').eval().to(device)
    elif args.device == "cuda":
        tokenizer, oringin_model, image_processor, context_len = load_pretrained_model(model_path, model_name, None)
        vision_tower = origin_model.vision_tower
        mm_projector = origin_model.mm_projector
        llm_model = origin_model.llm
    device = torch.device(args.device)
    llm_model.eval().to(device)
    llama = llm_model.model
    NUM_FRAMES = 1
    NUM_LAYERS = llm_cfg.num_hidden_layers
    HIDDEN_SIZE = llm_cfg.hidden_size
    SEQ_LENGTH = args.seq_length
    NUM_KEY_VALUE_HEADS = llm_cfg.num_key_value_heads
    HEAD_DIM = HIDDEN_SIZE // NUM_KEY_VALUE_HEADS
    VOCAB_SIZE = llm_cfg.vocab_size
    folder='tmp/onnx'

    test_net_with_mask()
    exit()

    # convert
    convert()
