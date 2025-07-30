# ==============================================================================
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import time
import argparse
import chat
import os
import torch
import numpy as np
import torch.nn.functional as F
from transformers import Qwen2_5OmniProcessor
from qwen_vl_utils import process_vision_info
import librosa, ffmpeg


class Qwen2_5O():

    def __init__(self, args):
        # devid
        self.device = args.devid

        # load model
        self.fps = 1.0
        self.model = chat.Qwen2_5O()
        self.model.init(self.device, args.model_path)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(args.config_path,
                                                              max_pixels=self.model.MAX_PIXELS,
                                                              min_pixels=64 * 28 * 28,
                                                              fps=self.fps)
        self.tokenizer = self.processor.tokenizer
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.ID_IMAGE = self.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        self.ID_VIDEO = self.tokenizer.convert_tokens_to_ids(self.processor.video_token)
        self.ID_AUDIO = self.tokenizer.convert_tokens_to_ids(self.processor.audio_token)
        self.ID_VISION_BOS = self.tokenizer.convert_tokens_to_ids(self.processor.vision_bos_token)
        self.ID_VISION_EOS = self.tokenizer.convert_tokens_to_ids(self.processor.vision_eos_token)
        self.ID_AUDIO_BOS = self.tokenizer.convert_tokens_to_ids(self.processor.audio_bos_token)
        self.ID_AUDIO_EOS = self.tokenizer.convert_tokens_to_ids(self.processor.audio_eos_token)

        self.spatial_merge_size = 2
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.tokens_per_second = 2
        self.support_history = self.model.support_history
        self.max_posid = 0
        self.history_max_posid = 0
        # yapf: disable
        self.system_prompt = {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
            }]}
        # yapf: enable

    def text_message(self):
        # yapf: disable
        messages = [
            {"role": "user", "content": [{"type": "text", "text": self.input_str}]}
        ]
        # yapf: enable
        return messages

    def image_message(self, path):
        # yapf: disable
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": path}
            ],
        }]
        if self.input_str != "":
            messages[0]["content"].append({"type": "text", "text": self.input_str})
        # yapf: enable
        return messages

    def video_message(self, path):
        # yapf: disable
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": path, "fps": self.fps,
                 "min_pixels": 64 * 28 * 28, "max_pixels": self.model.MAX_PIXELS // 4}
            ],
        }]
        # yapf: enable
        if self.input_str != "":
            messages[0]["content"].append({"type": "text", "text": self.input_str})
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

    def has_audio(self, media_path):
        info = ffmpeg.probe(media_path)
        return any(s.get("codec_type") == "audio" for s in info["streams"])

    def process(self, messages, media_path="", media_type=None):
        text = self.processor.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True)
        if media_type == "text":
            return self.processor(text=text, return_tensors="pt", padding=True)
        print(f"Processing {media_path} ......")
        images, videos = process_vision_info(messages)
        audios = None
        if media_type == "video" and self.has_audio(media_path):
            y, _ = librosa.load(media_path, sr=16000)
            y = y[:(len(y) // 16000) * 16000]
            audios = [y]
        inputs = self.processor(text=text,
                                audio=audios,
                                images=images,
                                videos=videos,
                                return_tensors="pt",
                                padding=True,
                                use_audio_in_video=(audios is not None))
        print("Inputs processed successfully.")
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

    def image_inference(self, inputs):
        vit_offset = int(torch.where(inputs.input_ids == self.ID_IMAGE)[1][0])
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
        self.model.forward_vit(hidden_states.numpy(), position_ids.numpy(), full_mask.numpy(),
                               window_mask.numpy(), grid_thw.numpy(), reverse_indices.numpy(),
                               vit_offset, False)

    def video_inference(self, inputs, is_video_audio=False):
        vit_offset = int(torch.where(
            inputs.input_ids == self.ID_VIDEO)[1][0]) if not is_video_audio else 0
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
            self.model.forward_vit(hidden_states.numpy(), position_ids.numpy(), full_mask.numpy(),
                                   window_mask.numpy(), grid_thw.numpy(), reverse_indices.numpy(),
                                   vit_offset, is_video_audio)
            vit_offset += seq_len // 4
            t_offset += t_i

    def video_audio_inference(self, inputs):
        start_offset = int(torch.where(inputs.input_ids == self.ID_VIDEO)[1][0])
        end_offset = int(torch.where(inputs.input_ids == self.ID_AUDIO_EOS)[1][0])
        t, h, w = inputs.video_grid_thw.flatten().tolist()
        total_vit_tokens = int(t * h * w // (self.spatial_merge_size**2))
        vit_tokens_per_chunk = int(self.fps * h * w // (self.spatial_merge_size**2))
        audio_tokens_per_chunk = 50  # 25 per second * 2 seconds
        tokens_per_chunk = vit_tokens_per_chunk + audio_tokens_per_chunk
        n_times = int(t // self.fps)
        # ========= audio inference ===============================

        audio_dims = inputs.feature_attention_mask.sum().item()
        audio_features = inputs.input_features[:, :, :audio_dims].reshape(128, -1,
                                                                          200).transpose(0, 1)
        total_audio_tokens = audio_features.shape[0] * audio_tokens_per_chunk
        audio_offset = start_offset + vit_tokens_per_chunk
        if audio_offset != int(torch.where(inputs.input_ids == self.ID_AUDIO)[1][0]):
            raise RuntimeError(
                f"Audio BOS token offset {audio_offset} does not match expected offset")
        if total_audio_tokens // audio_tokens_per_chunk != n_times:
            raise RuntimeError(
                f"Total audio tokens {total_audio_tokens} is not divisible by audio tokens per chunk {audio_tokens_per_chunk}"
            )
        audio_offset_list = [i * tokens_per_chunk + audio_offset for i in range(n_times)]
        if (audio_features.shape[0] != n_times):
            raise RuntimeError(
                f"Audio features shape {audio_features.shape[0]} does not match expected {n_times}")
        # do audio inference
        self.model.forward_audio(audio_features.numpy(), np.array(audio_offset_list,
                                                                  dtype=np.int32))
        # ========== video inference ===============================
        total_tokens = total_vit_tokens + total_audio_tokens
        if total_tokens != end_offset - start_offset:
            raise RuntimeError(
                f"Total tokens {total_tokens} does not match expected {end_offset - start_offset}")
        vit_offset = start_offset
        vit_offset_list = [i * tokens_per_chunk + vit_offset for i in range(n_times)]
        self.video_inference(inputs, is_video_audio=True)
        self.model.video_sync(vit_tokens_per_chunk, np.array(vit_offset_list, dtype=np.int32))

    def get_chunked_index(self, token_indices: torch.Tensor, tokens_per_chunk: int,
                          remove_index: int) -> list[tuple[int, int]]:

        def _iter():
            i, start_idx = 0, 0  # skip bos token
            current_chunk = 1
            while i < len(token_indices):  # skip eos token
                if token_indices[i] - remove_index >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))

        return list(_iter())

    def get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: list[int],
        grid_hs: list[int],
        grid_ws: list[int],
    ):
        llm_pos_ids_list = []
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1,
                                                                 llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h,
                                                                 -1).flatten()
        t_index = torch.Tensor(t_index).view(-1,
                                             1).expand(-1,
                                                       llm_grid_h * llm_grid_w).flatten().long()
        _llm_pos_ids = torch.stack([t_index, h_index, w_index])
        llm_pos_ids_list.append(_llm_pos_ids + start_idx)  # + 1 ) # 12.09 by malinhan
        llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
        return llm_pos_ids

    def get_rope_index(
        self,
        input_ids,
        image_grid_thw,
        video_grid_thw,
        attention_mask,
        audio_seqlens,
        second_per_grids,
    ) -> torch.Tensor:
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.ID_IMAGE
        video_token_id = self.ID_VIDEO
        audio_token_id = self.ID_AUDIO
        vision_start_token_id = self.ID_VISION_BOS
        audio_start_token_id = self.ID_AUDIO_BOS
        position_id_per_seconds = 25
        seconds_per_chunk = 2

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(3,
                                      input_ids.shape[0],
                                      input_ids.shape[1],
                                      dtype=input_ids.dtype)
            image_idx, video_idx, audio_idx = 0, 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums, audio_nums = 0, 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                audio_nums = torch.sum(input_ids == audio_start_token_id)
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == audio_start_token_id).sum()

                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
                multimodal_nums = (image_nums + audio_nums)
                for _ in range(multimodal_nums):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio = input_tokens.index(audio_token_id, st)
                    else:
                        ed_audio = len(input_tokens) + 1
                    min_ed = min(ed_image, ed_video, ed_audio)
                    if min_ed == ed_audio:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                                llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(
                                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(
                            torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                        llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(
                            torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + audio_len + eos_len
                        audio_idx += 1
                        remain_audios -= 1

                    elif min_ed == ed_image:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                                llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(
                                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(
                            torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).long()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(st_idx, image_idx,
                                                                      spatial_merge_size, t_index,
                                                                      grid_hs, grid_ws)
                        image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(
                            torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + image_len + eos_len
                        image_idx += 1
                        remain_images -= 1

                    elif min_ed == ed_video:
                        text_len = min_ed - st - 2
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                                llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(
                                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(
                            torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(
                            torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                        audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3,
                                                                                       -1) + st_idx
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (torch.arange(grid_t) * second_per_grids[video_idx].float() *
                                   position_id_per_seconds).long()
                        video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws)

                        t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        video_chunk_indexes = self.get_chunked_index(video_llm_pos_ids[0],
                                                                     t_ntoken_per_chunk, st_idx)
                        audio_chunk_indexes = self.get_chunked_index(audio_llm_pos_ids[0],
                                                                     t_ntoken_per_chunk, st_idx)
                        sub_len = 0
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            video_chunk_index = video_chunk_indexes[j] if j < len(
                                video_chunk_indexes) else None
                            audio_chunk_index = audio_chunk_indexes[j] if j < len(
                                audio_chunk_indexes) else None
                            if video_chunk_index is not None:
                                sub_len += video_chunk_index[1] - video_chunk_index[0]

                                llm_pos_ids_list.append(
                                    video_llm_pos_ids[:, video_chunk_index[0]:video_chunk_index[1]])
                            if audio_chunk_index is not None:
                                sub_len += audio_chunk_index[1] - audio_chunk_index[0]

                                llm_pos_ids_list.append(
                                    audio_llm_pos_ids[:, audio_chunk_index[0]:audio_chunk_index[1]])
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(
                            torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(
                            torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2

                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions

            return position_ids
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
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
            if media_path == "" and self.input_str == "":
                print("Error: No input, try again!!")
                continue
            elif media_path == "":
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
            messages.insert(0, self.system_prompt)
            inputs = self.process(messages, media_path, media_type)
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
            has_audio = False
            if 'feature_attention_mask' in inputs:
                audio_feature_lengths = torch.sum(inputs.feature_attention_mask, dim=1)
                has_audio = True
            else:
                audio_feature_lengths = None
            if media_type == "image":
                image_grid_thw = inputs.image_grid_thw
            else:
                image_grid_thw = None
            if media_type == "video":
                video_grid_thw = inputs.video_grid_thw
                video_second_per_grid = inputs.video_second_per_grid
            else:
                video_grid_thw = None
                video_second_per_grid = None
            position_ids = self.get_rope_index(inputs.input_ids, image_grid_thw, video_grid_thw,
                                               inputs.attention_mask, audio_feature_lengths,
                                               video_second_per_grid)
            self.max_posid = int(position_ids.max())
            print("\nAnswer:")

            # Chat
            first_start = time.time()
            self.model.forward_embed(inputs.input_ids.squeeze(0).tolist())
            if media_type == "image":
                self.image_inference(inputs)
            elif media_type == "video":
                if not has_audio:
                    self.video_inference(inputs)
                else:
                    self.video_audio_inference(inputs)

            token = self.forward_prefill(position_ids.numpy())
            first_end = time.time()
            tok_num = 0
            # Following tokens
            full_word_tokens = []
            text = ""
            while token not in [self.ID_IM_END] and self.model.history_length < self.model.SEQLEN:
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
    model = Qwen2_5O(args)
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
