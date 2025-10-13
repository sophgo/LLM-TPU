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
from qwen_vl_utils import process_vision_info
import chat
import os
import torch
import numpy as np
import torch.nn.functional as F


class Qwen3_VL():

    def __init__(self, args):
        # devid
        self.device = args.devid

        # load model
        self.model = chat.Qwen3_VL()
        self.model.init(self.device, args.model_path)
        self.processor = AutoProcessor.from_pretrained(args.config_path,
                                                       trust_remote_code=True,
                                                       max_pixels=self.model.MAX_PIXELS,
                                                       min_pixels=64 * 32 * 32)
        self.tokenizer = self.processor.tokenizer
        self.ID_END = self.tokenizer.convert_tokens_to_ids("<|end|>")
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.ID_IMAGE_PAD = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.ID_VIDEO_PAD = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.ID_VISION_START = self.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        self.spatial_merge_size = 2
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.tokens_per_second = 2
        self.support_history = self.model.support_history
        self.max_posid = 0
        self.history_max_posid = 0

    def text_message(self):
        # yapf: disable
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": self.input_str}],
        }]
        # yapf: enable
        return messages

    def image_message(self, path):
        # yapf: disable
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": path},
                {"type": "text", "text": self.input_str},
            ],
        }]
        # yapf: enable
        return messages

    def video_message(self, path):
        # yapf: disable
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": path, "fps": 1.0,
                 "min_pixels": 64 * 32 * 32, "max_pixels": self.model.MAX_PIXELS // 4},
                {"type": "text", "text": self.input_str},
            ],
        }]
        # yapf: enable
        return messages

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

    def process(self, messages):
        text = self.processor.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return inputs

    def rot_pos(self, grid_thw):
        merge_size = self.spatial_merge_size
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.int32)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h)  # block row indices
            block_cols = torch.arange(merged_w)  # block col indices
            intra_row = torch.arange(merge_size)  # intra-block row offsets
            intra_col = torch.arange(merge_size)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset:offset + num_tokens] = coords
            offset += num_tokens

            # lookup rotary embeddings
        return pos_ids

    def fast_pos_embed_interpolate(self, grid_thw):
        t, h, w = grid_thw
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.int32)
        weight_tensor = torch.tensor(weight_list, dtype=torch.float32)
        msize = self.spatial_merge_size
        idx_tensor = idx_tensor.view(4 * t, h // msize, msize, w // msize,
                                     msize).permute(0, 1, 3, 2, 4).reshape(4, t * h * w)
        weight_tensor = weight_tensor.view(4 * t, h // msize, msize, w // msize,
                                           msize).permute(0, 1, 3, 2, 4).reshape(4, t * h * w)

        return idx_tensor, weight_tensor

    def vit_process_image(self, inputs, vit_offset):
        grid_thw = inputs.image_grid_thw
        hidden_states = inputs.pixel_values
        position_ids = self.rot_pos(grid_thw)
        pos_ids, pos_weights = self.fast_pos_embed_interpolate(grid_thw.tolist())
        self.model.forward_vit(hidden_states.numpy(), position_ids.numpy(), pos_ids.numpy(),
                               pos_weights.numpy(), grid_thw.numpy(), vit_offset)

    def vit_process_video(self, inputs, vit_offset):
        t, h, w = inputs.video_grid_thw.flatten().tolist()
        grid_thw = torch.tensor([[1, h, w]], dtype=torch.int32)
        position_ids = self.rot_pos(grid_thw)
        pos_ids, pos_weights = self.fast_pos_embed_interpolate(grid_thw.tolist())
        for i in range(t):
            hidden_states = inputs.pixel_values_videos[(i * h * w):((i + 1) * h * w), :]
            self.model.forward_vit(hidden_states.numpy(), position_ids.numpy(), pos_ids.numpy(),
                                   pos_weights.numpy(), grid_thw.numpy(), vit_offset)
            vit_offset += h * w // 4

    def get_rope_index(self, input_ids: torch.LongTensor, grid_thw: torch.LongTensor,
                       pad_id: int) -> torch.Tensor:
        total_input_ids = input_ids
        attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype)
        image_index = 0
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums = 0
            vision_start_indices = torch.argwhere(input_ids == self.ID_VISION_START).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == pad_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images = image_nums
            for _ in range(image_nums):
                if pad_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(pad_id, st)
                else:
                    ed_image = len(input_tokens) + 1

                t, h, w = (
                    grid_thw[image_index][0],
                    grid_thw[image_index][1],
                    grid_thw[image_index][2],
                )
                second_per_grid_t = 0 if pad_id == self.ID_IMAGE_PAD else 1
                image_index += 1
                remain_images -= 1
                ed = ed_image

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // self.spatial_merge_size,
                    w.item() // self.spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * self.tokens_per_second

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = torch.arange(llm_grid_h).view(1, -1,
                                                        1).expand(llm_grid_t, -1,
                                                                  llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1,
                                                        -1).expand(llm_grid_t, llm_grid_h,
                                                                   -1).flatten()
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions
        return position_ids

    def forward_prefill(self, position_ids):
        if self.model.history_length == 0 or not self.support_history:
            self.history_max_posid = 0
            return self.model.forward_first(position_ids)
        self.max_posid += self.history_max_posid
        position_ids = position_ids + self.history_max_posid
        return self.model.forward_first(position_ids)

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
            if self.input_str in ["clear", "new", "c"]:
                print("New chat session created.")
                self.model.clear_history()
                self.history_max_posid = 0
                continue

            media_path = input("\nImage or Video Path: ")
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
                    messages = self.image_message(media_path)
                elif media_type == "video":
                    messages = self.video_message(media_path)
                else:
                    print("Unsupported media type: {}".format(media_path))
                    continue

            inputs = self.process(messages)
            token_len = inputs.input_ids.numel()
            if token_len > self.model.MAX_INPUT_LENGTH:
                print(
                    "Error: The maximum question length should be shorter than {} but we get {} instead."
                    .format(self.model.MAX_INPUT_LENGTH, token_len))
                continue
            if self.support_history:
                if (token_len + self.model.history_length > self.model.SEQLEN - 128) or \
                (self.model.history_length > self.model.PREFILL_KV_LENGTH):
                    print("Warning: History is full and clear it to continue.")
                    self.model.clear_history()
                    self.history_max_posid = 0
            print("\nAnswer:")

            # Chat
            first_start = time.time()
            self.model.forward_embed(inputs.input_ids.squeeze(0).tolist())
            if media_type == "image":
                vit_token_list = torch.where(inputs.input_ids == self.ID_VISION_START)[1].tolist()
                vit_offset = vit_token_list[0] + 1
                self.vit_process_image(inputs, vit_offset)
                position_ids = self.get_rope_index(inputs.input_ids, inputs.image_grid_thw,
                                                   self.ID_IMAGE_PAD)
                self.max_posid = int(position_ids.max())
                token = self.forward_prefill(position_ids.numpy())
            elif media_type == "video":
                vit_token_list = torch.where(inputs.input_ids == self.ID_VISION_START)[1].tolist()
                vit_offset = vit_token_list[0] + 1
                self.vit_process_video(inputs, vit_offset)
                position_ids = self.get_rope_index(inputs.input_ids, inputs.video_grid_thw,
                                                   self.ID_VIDEO_PAD)
                self.max_posid = int(position_ids.max())
                token = self.forward_prefill(position_ids.numpy())
            else:
                position_ids = 3 * [i for i in range(token_len)]
                self.max_posid = token_len - 1
                token = self.forward_prefill(np.array(position_ids, dtype=np.int32))
            first_end = time.time()
            tok_num = 0
            # Following tokens
            full_word_tokens = []
            text = ""
            while token not in [self.ID_IM_END, self.ID_END
                                ] and self.model.history_length < self.model.SEQLEN:
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
                self.max_posid += 1
                position_ids = np.array([self.max_posid, self.max_posid, self.max_posid],
                                        dtype=np.int32)
                token = self.model.forward_next(position_ids)
                tok_num += 1
            self.history_max_posid = self.max_posid + 2
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")


def main(args):
    model = Qwen3_VL(args)
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
