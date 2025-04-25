import time
import argparse
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import chat
import os
import torch
import torch.nn.functional as F


class Qwen2_5VL():

    def __init__(self, args):
        # devid
        self.device = args.devid

        # load model
        self.model = chat.Qwen2_5VL()
        self.model.init(self.device, args.model_path)
        self.processor = AutoProcessor.from_pretrained(args.config_path,
                                                       trust_remote_code=True,
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
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": self.input_str
                },
            ],
        }]
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
                {"type": "video", "video": path, "fps": 1.0},
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

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = 4

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // 2,
                grid_w // 2,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

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

    def vit_process(self, inputs, vit_offset):
        grid_thw = inputs.image_grid_thw
        hidden_states = inputs.pixel_values
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        seq_len, _ = hidden_states.shape
        # reorder hidden_states
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit,
                                              self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        # reorder position_ids
        position_ids = self.rot_pos(grid_thw)
        position_ids = position_ids.reshape(seq_len // self.spatial_merge_unit,
                                            self.spatial_merge_unit, -1)
        position_ids = position_ids[window_index, :, :]
        position_ids = position_ids.reshape(seq_len, -1)
        # cu_seqlens
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cumsum(
                                                 dim=0,
                                                 dtype=torch.int32,
                                             )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        full_mask = self.get_attn_mask(seq_len, cu_seqlens)
        window_mask = self.get_attn_mask(seq_len, cu_window_seqlens)
        reverse_indices = torch.argsort(window_index)
        self.model.forward_vit(hidden_states.flatten().tolist(),
                               position_ids.flatten().tolist(),
                               full_mask.flatten().tolist(),
                               window_mask.flatten().tolist(),
                               grid_thw.flatten().tolist(),
                               reverse_indices.flatten().tolist(), vit_offset)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: torch.LongTensor,
    ) -> torch.Tensor:
        total_input_ids = input_ids
        attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index = 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums = 0
            vision_start_indices = torch.argwhere(input_ids == self.ID_VISION_START).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == self.ID_IMAGE_PAD).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images = image_nums
            for _ in range(image_nums):
                if self.ID_IMAGE_PAD in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(self.ID_IMAGE_PAD, st)
                else:
                    ed_image = len(input_tokens) + 1

                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                second_per_grid_t = 0
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
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
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
            if not os.path.exists(media_path):
                print("Can't find image or video: {}".format(media_path))
                continue
            media_type = self.get_media_type(media_path)
            if media_type == "image":
                messages = self.image_message(media_path)
            else:
                messages = self.video_message(media_path)
            inputs = self.process(messages)
            print("\nAnswer:")

            # Chat
            first_start = time.time()
            if media_type == "image":
                vit_token_list = torch.where(inputs.input_ids == self.ID_IMAGE_PAD)[1].tolist()
                vit_offset = vit_token_list[0]
                self.model.forward_embed(inputs.input_ids.squeeze(0).tolist())
                self.vit_process(inputs, vit_offset)
                position_ids = self.get_rope_index(inputs.input_ids, inputs.image_grid_thw)
                max_posid = int(position_ids.max())
                token = self.model.forward_first(position_ids.flatten().tolist())
            elif media_type == "video":
                vit_token_list = torch.where(inputs.input_ids == self.ID_VIDEO_PAD)[1].tolist()
                vit_offset = vit_token_list[0]
                token = self.model.forward_first(
                    inputs.input_ids.squeeze(0).tolist(),
                    inputs.pixel_values_videos.flatten().tolist(),
                    inputs.video_grid_thw.squeeze(0).tolist(), vit_offset)
            else:
                empty = []
                token = self.model.forward_first(
                    inputs.input_ids.squeeze(0).tolist(), empty, [], 0, 0)
            first_end = time.time()
            tok_num = 1
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

                token = self.model.forward_next([max_posid, max_posid, max_posid])
                tok_num += 1
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")


def main(args):
    model = Qwen2_5VL(args)
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
