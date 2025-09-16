# ==============================================================================
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import time
import argparse
from transformers import AutoProcessor
import chat
import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoConfig
from PIL import Image
import json
from copy import deepcopy
from decord import VideoReader, cpu

MAX_NUM_FRAMES = 64

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

class MiniCPMV4():

    def __init__(self, args):
        # devid
        self.device = args.devid

        # load model
        self.model = chat.MiniCPMV4()
        self.model.init(self.device, args.model_path)
        self.processor = AutoProcessor.from_pretrained(args.config_path,
                                                       trust_remote_code=True,
                                                       size=None,
                                                       max_pixels=self.model.MAX_PIXELS,
                                                       min_pixels=64 * 28 * 28)
        self.tokenizer = self.processor.tokenizer
        self.ID_END = self.tokenizer.convert_tokens_to_ids('</s>')
        self.ID_START = self.tokenizer.convert_tokens_to_ids('<s>')
        self.ID_IMAGE_START = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
        self.ID_IMAGE_END = self.tokenizer.convert_tokens_to_ids('<|im_end|>')

        self.vision_config = AutoConfig.from_pretrained(args.config_path, trust_remote_code=True).vision_config
        self.hidden_size = self.vision_config.hidden_size
        self.num_patches_per_side = self.vision_config.image_size // self.vision_config.patch_size

    def text_message(self):
        # yapf: disable
        messages = [{
            "role": "user",
            "content": [
                self.input_str
            ],
        }]
        # yapf: enable
        return messages

    def image_message(self, path):
        # yapf: disable
        image = Image.open(path).convert('RGB')
        messages = [{
            "role": "user",
            "content": [
                image,
                self.input_str
            ],
        }]
        # yapf: enable
        return image, messages

    def video_message(self, path):
        frames = encode_video(path)
        # yapf: disable
        messages = [{
            "role": "user",
            "content": frames + [self.input_str],
        }]
        # yapf: enable
        return frames, messages

    def get_media_type(self, file_path):
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext in image_exts:
            return "image"
        if ext in video_exts:
            return "video"
        raise RuntimeError(f"Unsupported media type: {ext}")

    def process(self, image_list, msgs, media_type):
        prompts_lists = []
        input_images_lists = []
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        copy_msgs = deepcopy(msgs)

        if image_list is not None and isinstance(copy_msgs[0]["content"], str):
            copy_msgs[0]["content"] = [image_list, copy_msgs[0]["content"]]

        images = []
        for i, msg in enumerate(copy_msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["user", "assistant"]
            if i == 0:
                assert role == "user", "The role of first msg should be user"
            if isinstance(content, str):
                content = [content]
            cur_msgs = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur_msgs.append("(<image>./</image>)")
                elif isinstance(c, str):
                    cur_msgs.append(c)
            msg["content"] = "\n".join(cur_msgs)

        prompt = self.processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True)
        prompts_lists.append(prompt)
        input_images_lists.append(images)

        max_slice_nums = None
        use_image_id = None
        if media_type == "video":
            max_slice_nums = 2
            use_image_id = False
        max_inp_length = 32768
        inputs = self.processor(prompts_lists,
                                input_images_lists, 
                                max_slice_nums=max_slice_nums,
                                use_image_id=use_image_id,
                                return_tensors="pt",
                                max_length=max_inp_length)
        return inputs

    def get_attn_mask(self, seq_length, cu_seqlens):
        attention_mask = torch.full([1, seq_length, seq_length], -10000.0, dtype=torch.float32)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
                           cu_seqlens[i - 1]:cu_seqlens[i]] = 0
        return attention_mask.numpy()

    # modeling_navit_siglip.py:SiglipVisionEmbeddings
    def get_position_ids(self, tgt_size: np.array):
        tgt_size = tgt_size.tolist()
        position_ids = torch.full(
            size=(
                tgt_size[0] * tgt_size[1],
            ),
            fill_value=0,
        )

        nb_patches_h = tgt_size[0]
        nb_patches_w = tgt_size[1]

        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
        fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
        pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
        position_ids = pos_ids

        return position_ids.numpy()
    
    # resampler.py:Resampler
    def get_pos_embed_ids(self, tgt_size):
        pos_embed_ids = []
        height, width = tgt_size
        for h in range(height):
            for w in range(width):
                pos_id = h * self.num_patches_per_side + w
                pos_embed_ids.append(pos_id)
        
        pos_embed_ids = np.array(pos_embed_ids, dtype=np.int32)
        return pos_embed_ids

    def vit_process_image(self, inputs):
        pixel_values = inputs.pixel_values[0] # batch = 1
        image_bound = inputs.image_bound[0]
        tgt_sizes = inputs.tgt_sizes[0]
        for i, pixel_value in  enumerate(pixel_values):
            tgt_size = tgt_sizes[i].numpy()
            pixel_value = pixel_value.numpy()
            pixel_value = np.transpose(pixel_value.reshape(3, 14, -1, 14), [2, 0, 1, 3])
            hidden_states = np.ascontiguousarray(pixel_value.reshape(-1, 3 * 14 * 14))
            cu_seqlens = torch.tensor([tgt_size[0] * tgt_size[1]], dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
            seq_len, _ = hidden_states.shape
            full_mask = self.get_attn_mask(seq_len, cu_seqlens)
            vit_offset = image_bound[i][0].item()
            position_ids = self.get_position_ids(tgt_size)
            pos_embed_ids = self.get_pos_embed_ids(tgt_size)
            self.model.forward_vit(hidden_states, position_ids, full_mask, pos_embed_ids,
                                    tgt_size, vit_offset)

    def vit_process_video(self, inputs):
        self.vit_process_image(inputs)

    def chat(self):
        """
        Start a chat session.
        """
        # Instruct
        print("""\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
=================================================================""")
        # Stop Chatting with "exit" input
        while True:
            self.input_str = input("\nQuestion: ")
            # Quit
            if self.input_str in ["exit", "q", "quit"]:
                break

            media_path = input("\nImage or Video Path: ")

            images = None
            media_path = media_path.strip()
            if media_path == "":
                messages = self.text_message()
                media_type = "text"
            elif not os.path.exists(media_path):
                print("Can't find image or video: {}".format(media_path))
                continue
            else:
                media_type = self.get_media_type(media_path)
                if media_type == "image":
                    images, messages = self.image_message(media_path)
                elif media_type == "video":
                    images, messages = self.video_message(media_path)
                else:
                    print("Unsupported media type: {}".format(media_path))
                    continue
            inputs = self.process(images, messages, media_type)
            token_len = inputs.input_ids.numel()
            if token_len >= self.model.SEQLEN - 128:
                print(
                    "The maximum question length should be shorter than {} but we get {} instead.".
                    format(self.model.SEQLEN, token_len))
                continue
            print("\nAnswer:")

            # Chat
            first_start = time.time()
            self.model.forward_embed(inputs.input_ids.squeeze(0).tolist())
            if media_type == "image":
                self.vit_process_image(inputs)
                position_ids = np.arange(inputs.input_ids.shape[1])
                max_posid = int(position_ids.max())
                token = self.model.forward_first(position_ids)
            elif media_type == "video":
                self.vit_process_video(inputs)
                position_ids = np.arange(inputs.input_ids.shape[1])
                max_posid = int(position_ids.max())
                token = self.model.forward_first(position_ids)
            else:
                position_ids = np.arange(inputs.input_ids.shape[1])
                max_posid = token_len - 1
                token = self.model.forward_first(position_ids)
            first_end = time.time()
            tok_num = 0
            # Following tokens
            full_word_tokens = []
            text = ""
            while token not in [self.ID_IMAGE_END, self.ID_END]:
                if self.model.token_length >= self.model.SEQLEN:
                    print(f"\ntoken_length has been {self.model.SEQLEN}, terminated!")
                    break
                full_word_tokens.append(token)
                word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
                if "�" not in word:
                    if len(full_word_tokens) == 1:
                        pre_word = word
                        word = self.tokenizer.decode([token, token],
                                                     skip_special_tokens=True)[len(pre_word):]
                    text += word
                    print(word, flush=True, end="")
                    full_word_tokens = []
                max_posid += 1
                position_ids = np.array([max_posid], dtype=np.int32)
                token = self.model.forward_next(position_ids)
                tok_num += 1
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")


def main(args):
    model = MiniCPMV4(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="../config",
                        help='path to the processor file')
    parser.add_argument('-d', '--devid', type=int, default=0, help='device ID to use')
    # yapf: enable
    args = parser.parse_args()
    main(args)
