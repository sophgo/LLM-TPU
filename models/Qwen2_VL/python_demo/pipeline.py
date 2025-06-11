import time
import argparse
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import chat
import os
import torch
import numpy as np
import torch.nn.functional as F


class Qwen2VL():

    def __init__(self, args):
        # devid
        self.device = args.devid

        # load model
        self.model = chat.Qwen2VL()
        self.model.init(self.device, args.model_path)
        self.processor = AutoProcessor.from_pretrained(args.config_path,
                                                       trust_remote_code=True,
                                                       size=None,
                                                       max_pixels=self.model.MAX_PIXELS,
                                                       min_pixels=256 * 28 * 28)
        self.tokenizer = self.processor.tokenizer
        self.ID_END = self.tokenizer.convert_tokens_to_ids("<|end|>")
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.ID_IMAGE_PAD = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.ID_VIDEO_PAD = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        self.ID_VISION_START = self.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        self.spatial_merge_size = 2
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.tokens_per_second = 2

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
                 "min_pixels": 64 * 28 * 28, "max_pixels": self.model.MAX_PIXELS // 4},
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

    def get_attn_mask(self, seq_length, cu_seqlens):
        attention_mask = torch.full([1, seq_length, seq_length], -10000.0, dtype=torch.float32)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
                           cu_seqlens[i - 1]:cu_seqlens[i]] = 0
        return attention_mask

    def rot_pos(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        return pos_ids

    def vit_process_image(self, inputs, vit_offset):
        grid_thw = inputs.image_grid_thw
        hidden_states = inputs.pixel_values
        seq_len, _ = hidden_states.shape
        position_ids = self.rot_pos(grid_thw)
        # cu_seqlens
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cumsum(
                                                 dim=0,
                                                 dtype=torch.int32,
                                             )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        full_mask = self.get_attn_mask(seq_len, cu_seqlens)
        self.model.forward_vit(hidden_states.numpy(), position_ids.numpy(), full_mask.numpy(),
                               grid_thw.numpy(), vit_offset)

    def vit_process_video(self, inputs, vit_offset):
        t, h, w = inputs.video_grid_thw.flatten().tolist()
        per_t = self.model.MAX_PATCHES // (h * w)
        t_list = []
        if per_t >= t:
            t_list = [t]
        else:
            t_list = [per_t] * (t // per_t) + ([t % per_t] if t % per_t else [])
        t_offset = 0
        for t_i in t_list:
            grid_thw = torch.tensor([[t_i, h, w]], dtype=torch.int32)
            hidden_states = inputs.pixel_values_videos[(t_offset * h * w):((t_offset + t_i) * h *
                                                                           w), :]
            seq_len, _ = hidden_states.shape
            # reorder position_ids
            position_ids = self.rot_pos(grid_thw)
            # cu_seqlens
            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                                 grid_thw[:, 0]).cumsum(
                                                     dim=0,
                                                     dtype=torch.int32,
                                                 )
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
            full_mask = self.get_attn_mask(seq_len, cu_seqlens)
            self.model.forward_vit(hidden_states.numpy(), position_ids.numpy(), full_mask.numpy(),
                                   grid_thw.numpy(), vit_offset)
            vit_offset += seq_len // 4
            t_offset += t_i

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

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h *
                                                                      llm_grid_w).flatten()

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
                vit_token_list = torch.where(inputs.input_ids == self.ID_IMAGE_PAD)[1].tolist()
                vit_offset = vit_token_list[0]
                self.vit_process_image(inputs, vit_offset)
                position_ids = self.get_rope_index(inputs.input_ids, inputs.image_grid_thw,
                                                   self.ID_IMAGE_PAD)
                max_posid = int(position_ids.max())
                token = self.model.forward_first(position_ids.numpy())
            elif media_type == "video":
                vit_token_list = torch.where(inputs.input_ids == self.ID_VIDEO_PAD)[1].tolist()
                vit_offset = vit_token_list[0]
                self.vit_process_video(inputs, vit_offset)
                position_ids = self.get_rope_index(inputs.input_ids, inputs.video_grid_thw,
                                                   self.ID_VIDEO_PAD)
                max_posid = int(position_ids.max())
                token = self.model.forward_first(position_ids.numpy())
            else:
                position_ids = 3 * [i for i in range(token_len)]
                max_posid = token_len - 1
                token = self.model.forward_first(position_ids)
            first_end = time.time()
            tok_num = 0
            # Following tokens
            full_word_tokens = []
            text = ""
            while token not in [self.ID_IM_END, self.ID_END
                                ] and self.model.token_length < self.model.SEQLEN:
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
                position_ids = np.array([max_posid, max_posid, max_posid], dtype=np.int32)
                token = self.model.forward_next(position_ids)
                tok_num += 1
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")


def main(args):
    model = Qwen2VL(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="config",
                        help='path to the processor file')
    parser.add_argument('-d', '--devid', type=int, default=0, help='device ID to use')
    # yapf: enable
    args = parser.parse_args()
    main(args)
