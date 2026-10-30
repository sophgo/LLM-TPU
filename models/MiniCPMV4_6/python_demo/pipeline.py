# ==============================================================================
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
#
# MiniCPM-V-4.6 / MiniCPM-V-4.7 unified pipeline.
#
# Both models share the same text backbone (qwen3_5_text) and the same ViT
# net I/O contract, so a single pipeline handles both.  The version is
# auto-detected from the config directory:
#
#   * MiniCPM-V-4.6 (`mrope_mode` absent, plain 3-D M-RoPE)
#   * MiniCPM-V-4.7 (`mrope_mode == "canvas"`, Canvas M-RoPE)
#
# The only functional difference is `get_rope_index()`: 4.7 anchors each
# <image>/<slice> span on a shared canvas and needs `mm_token_type_ids`
# (returning rope_deltas for decode), while 4.6 uses plain per-block 3-D
# positions.  Token ids are read from the model config when present (4.7)
# and fall back to the tokenizer (4.6); the values are identical.
#
# chat.cpp / CMakeLists.txt are shared (pybind class `MiniCPMV4_6`).

import time
import argparse
import warnings
from transformers import AutoProcessor, AutoConfig
import chat
import os
import sys
import torch
import numpy as np
import itertools

warnings.filterwarnings("ignore", message=".*torchcodec.*")


class MiniCPMV():

    def __init__(self, args):
        # devid
        self.device = args.devid

        # load model
        self.model = chat.MiniCPMV4_6()  # pybind class; net orchestration identical
        self.model.init(self.device, args.model_path)
        self.processor = AutoProcessor.from_pretrained(args.config_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        # Placeholder token IDs (ViT scatter + im_end detection)
        self.ID_IMAGE_PAD = self.tokenizer.convert_tokens_to_ids('<|image_pad|>')  # 248056
        self.ID_VIDEO_PAD = self.tokenizer.convert_tokens_to_ids('<|video_pad|>')  # 248057
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids('<|im_end|>')

        # Vision marker ids.  4.7 stores them in config.json; 4.6 does not, so
        # fall back to the tokenizer (the values are the same: <image>=248078,
        # <slice>=248088, newline=198).
        cfg = AutoConfig.from_pretrained(args.config_path)
        self.is_canvas = getattr(cfg, "mrope_mode", None) == "canvas"
        self.image_start_id = getattr(cfg, "image_start_id", None) \
            or self.tokenizer.convert_tokens_to_ids('<image>')      # 248078
        self.image_end_id = getattr(cfg, "image_end_id", None) \
            or self.tokenizer.convert_tokens_to_ids('</image>')     # 248079
        self.slice_start_id = getattr(cfg, "slice_start_id", None) \
            or self.tokenizer.convert_tokens_to_ids('<slice>')      # 248088
        self.slice_end_id = getattr(cfg, "slice_end_id", None) \
            or self.tokenizer.convert_tokens_to_ids('</slice>')     # 248089
        self.newline_id = getattr(cfg, "newline_id", None) or 198   # '\n'

        # ViT config
        self.downsample_mode = args.downsample_mode
        self.max_num_frames = args.max_num_frames
        self.max_slice_nums = args.max_slice_nums
        # 4x: Merger 2×2 → total 4x reduction; 16x: ViT Merger 2×2 + Merger 2×2 → total 16x
        self.spatial_merge_size = 2 if self.downsample_mode == "4x" else 4
        self.support_history = self.model.support_history
        self.num_grid_per_side = 70  # 980 / 14
        self.max_posid = 0
        self.history_max_posid = 0

    def __del__(self):
        if hasattr(self, 'model'):
            self.model.deinit()

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

    # ======================== Processor ========================

    def process(self, input_str, media_type, media_path=""):
        """Build messages and tokenize via processor."""
        if media_type == "text":
            content = [{"type": "text", "text": input_str}]
        elif media_type == "image":
            content = [{"type": "image", "image": media_path},
                       {"type": "text", "text": input_str}]
        else:  # video
            content = [{"type": "video", "video": media_path},
                       {"type": "text", "text": input_str}]
        messages = [{"role": "user", "content": content}]

        common = dict(tokenize=True, add_generation_prompt=True,
                      return_dict=True, return_tensors="pt")
        if media_type == "text":
            return self.processor.apply_chat_template(messages, **common)
        if media_type == "image":
            return self.processor.apply_chat_template(
                messages, **common,
                processor_kwargs=dict(
                    downsample_mode=self.downsample_mode,
                    max_slice_nums=self.max_slice_nums or 36))
        else:  # video
            return self.processor.apply_chat_template(
                messages, **common,
                processor_kwargs=dict(
                    downsample_mode=self.downsample_mode,
                    max_slice_nums=self.max_slice_nums or 1,
                    max_num_frames=self.max_num_frames,
                    stack_frames=1, use_image_id=False))

    # ======================== Index computation (ViT) ========================

    def compute_pos_ids(self, h_patches, w_patches):
        """MiniCPM-V bucketize position encoding."""
        boundaries = torch.arange(1 / self.num_grid_per_side, 1.0, 1 / self.num_grid_per_side)
        fractional_h = torch.arange(0, 1 - 1e-6, 1 / h_patches)
        fractional_w = torch.arange(0, 1 - 1e-6, 1 / w_patches)
        bucket_h = torch.bucketize(fractional_h, boundaries, right=True)
        bucket_w = torch.bucketize(fractional_w, boundaries, right=True)
        return (bucket_h[:, None] * self.num_grid_per_side + bucket_w).flatten()

    def compute_reorder_index(self, h, w):
        """2x2 spatial grouping reorder index for grid (h, w)."""
        n_out = (h // 2) * (w // 2)
        reorder = np.zeros(n_out * 4, dtype=np.int32)
        for mh in range(h // 2):
            for mw in range(w // 2):
                out_pos = (mh * (w // 2) + mw) * 4
                reorder[out_pos + 0] = (mh * 2 + 0) * w + (mw * 2 + 0)
                reorder[out_pos + 1] = (mh * 2 + 0) * w + (mw * 2 + 1)
                reorder[out_pos + 2] = (mh * 2 + 1) * w + (mw * 2 + 0)
                reorder[out_pos + 3] = (mh * 2 + 1) * w + (mw * 2 + 1)
        return reorder

    # ======================== ViT processing ========================

    def _find_vision_block_positions(self, ids):
        """Find start positions of vision blocks (<image> or <slice>) in token list."""
        positions = []
        for i, tid in enumerate(ids):
            if tid == self.image_start_id or tid == self.slice_start_id:
                positions.append(i)
        return positions

    def _vit_process(self, pixel_values, target_sizes, block_positions):
        """Process NaViT format vision blocks through ViT bmodel."""
        mode = self.downsample_mode
        assert len(block_positions) == len(target_sizes), \
            f"Found {len(block_positions)} vision blocks but {len(target_sizes)} target_sizes"

        col_offset = 0
        for idx in range(len(target_sizes)):
            h_p = int(target_sizes[idx, 0])
            w_p = int(target_sizes[idx, 1])
            n_patches = h_p * w_p

            slice_cols = n_patches * 14
            slice_pixel = pixel_values[:, :, :, col_offset:col_offset + slice_cols].numpy()
            col_offset += slice_cols

            pos_ids = self.compute_pos_ids(h_p, w_p).numpy()

            if mode == "4x":
                reorder_index = self.compute_reorder_index(h_p, w_p)
                window_index = np.zeros(1, dtype=np.int32)
                reverse_index = np.zeros(1, dtype=np.int32)
            else:  # 16x
                window_index = self.compute_reorder_index(h_p, w_p)
                reverse_idx = np.argsort(window_index).astype(np.int32)
                h_reduced, w_reduced = h_p // 2, w_p // 2
                reorder_index = self.compute_reorder_index(h_reduced, w_reduced)
                reverse_index = reverse_idx

            vit_offset = block_positions[idx]
            self.model.forward_vit(slice_pixel, pos_ids,
                                   reorder_index, window_index,
                                   reverse_index, mode, vit_offset + 1)

    def vit_process_image(self, inputs):
        """Process image slices through ViT."""
        ids = inputs.input_ids[0].tolist()
        block_positions = self._find_vision_block_positions(ids)
        self._vit_process(inputs.pixel_values, inputs.target_sizes, block_positions)

    def vit_process_video(self, inputs):
        """Process video frames through ViT."""
        ids = inputs.input_ids[0].tolist()
        block_positions = [i for i, tid in enumerate(ids) if tid == self.image_start_id]
        self._vit_process(inputs.pixel_values_videos, inputs.target_sizes_videos,
                          block_positions)

    # ======================== M-RoPE ========================

    def get_rope_index(self, input_ids, pad_id=None, mm_token_type_ids=None,
                       target_sizes=None, target_sizes_videos=None):
        """Dispatch to Canvas M-RoPE (4.7) or plain 3-D M-RoPE (4.6).

        Returns (position_ids, rope_deltas).  rope_deltas is 0 for 4.6.
        """
        if self.is_canvas:
            return self._get_rope_index_canvas(
                input_ids, mm_token_type_ids, target_sizes, target_sizes_videos)
        return self._get_rope_index_legacy(input_ids, target_sizes, pad_id), 0

    # ---------------- Plain 3-D M-RoPE (MiniCPM-V-4.6) ----------------

    def _get_rope_index_legacy(self, input_ids, target_sizes, pad_id):
        """Compute 3D MRoPE position IDs for the text model.

        MiniCPM token structure per image:
            <image> pad×N </image>          ← main block → target_sizes[i]
            <slice> pad×M </slice>          ← slice 0    → target_sizes[i+1]
            ... (may have \n between slice rows)
        Each block's <|image_pad|> tokens get 3D MRoPE position IDs based on
        that block's grid dimensions.
        """

        def _make_vision_pos_ids(t, h, w, merge_size, offset):
            """Generate 3D MRoPE position IDs for one vision block."""
            llm_grid_h = h // merge_size
            llm_grid_w = w // merge_size
            t_index = torch.arange(t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(t, llm_grid_h, -1).flatten()
            return torch.stack([t_index, h_index, w_index]) + offset

        total_input_ids = input_ids
        position_ids = torch.ones(3, input_ids.shape[0], input_ids.shape[1])
        image_index = 0

        # Find all contiguous pad_id block start positions
        input_tokens = total_input_ids[0].tolist()
        pad_positions = []
        for j in range(len(input_tokens)):
            if input_tokens[j] == pad_id and (j == 0 or input_tokens[j - 1] != pad_id):
                pad_positions.append(j)
        pad_block_idx = 0

        for i, input_ids in enumerate(total_input_ids):
            vision_start_indices = torch.argwhere(
                input_ids == self.image_start_id).squeeze(1)
            image_nums = len(vision_start_indices)
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0

            for img_idx in range(image_nums):
                # Count total pad blocks for this image (1 main + N slices)
                n_slices = 0
                search_start = st
                search_end = vision_start_indices[img_idx + 1].item() \
                    if img_idx + 1 < image_nums else len(input_tokens)
                for k in range(search_start, search_end):
                    if input_tokens[k] == self.slice_start_id:
                        n_slices += 1
                n_blocks = 1 + n_slices

                for block_idx in range(n_blocks):
                    ed = pad_positions[pad_block_idx]
                    pad_block_idx += 1
                    # Next pad block start, or end of sequence for last block
                    ed_next = pad_positions[pad_block_idx] \
                        if pad_block_idx < len(pad_positions) else len(input_tokens)

                    h = target_sizes[image_index][0].item()
                    w = target_sizes[image_index][1].item()
                    t = 1
                    image_index += 1

                    llm_grid_h = h // self.spatial_merge_size
                    llm_grid_w = w // self.spatial_merge_size
                    n_vision = t * llm_grid_h * llm_grid_w

                    # Pre-vision text: tokens from st to first pad (exclusive)
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    if text_len > 0:
                        llm_pos_ids_list.append(
                            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    llm_pos_ids_list.append(
                        _make_vision_pos_ids(t, h, w, self.spatial_merge_size,
                                             text_len + st_idx))

                    # Post-vision text: tokens between end of this vision block
                    # and the start of the next pad block (e.g. </image>,
                    # <image_id>, <slice>, \n).  For the last block this
                    # captures all trailing tokens through end of sequence.
                    post_text_len = ed_next - ed - n_vision
                    if post_text_len > 0:
                        post_st_idx = llm_pos_ids_list[-1].max() + 1
                        llm_pos_ids_list.append(
                            torch.arange(post_text_len).view(1, -1).expand(3, -1) +
                            post_st_idx)
                    st = ed_next

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, :] = llm_positions
        return position_ids.to(torch.int32)

    # ---------------- Canvas M-RoPE (MiniCPM-V-4.7) ----------------
    # Direct port of modelling_minicpmv4_7.py host-side Python.

    @staticmethod
    def _frame_end_idx(last_crop_end_idx, input_ids, markers, limit):
        end_idx = last_crop_end_idx
        while end_idx < limit and input_ids[end_idx] in markers:
            end_idx += 1
        return end_idx

    def _group_visual_frames(self, input_ids, mm_token_type_ids):
        """Split one sequence into visual groups of (thumbnail + slices) frames.

        Returns a list of (group_start_idx, group_end_idx, frames); each frame is
        (modality, thumb_start, thumb_end, frame_end, slices) with slices a list
        of (start, end) crop spans.
        """
        markers = (self.slice_start_id, self.slice_end_id,
                   self.image_start_id, self.image_end_id, self.newline_id)
        slice_start_id = markers[0]
        seq_len = len(input_ids)
        groups = []
        open_slices = None
        previous_end_idx = 0
        for modality, run in itertools.groupby(enumerate(mm_token_type_ids), lambda it: it[1]):
            if modality == 0:
                continue
            run = list(run)
            crop = (run[0][0], run[-1][0] + 1)
            if open_slices is not None and input_ids[crop[0] - 1] == slice_start_id:
                open_slices.append(crop)
            else:
                open_slices = []
                frame = (modality, crop[0], crop[1], open_slices)
                gap = input_ids[previous_end_idx: crop[0] - 1]
                adjacent = bool(groups) and all(t in markers for t in gap)
                if adjacent:
                    groups[-1][1].append(frame)
                else:
                    groups.append((crop[0] - 1, [frame]))
            previous_end_idx = crop[1]

        visual_groups = []
        for group_start_idx, frames in groups:
            frames = [
                (modality, ts, te,
                 self._frame_end_idx(slices[-1][1] if slices else te, input_ids, markers, seq_len),
                 slices)
                for modality, ts, te, slices in frames
            ]
            group_end_idx = frames[-1][3]
            visual_groups.append((group_start_idx, group_end_idx, frames))
        return visual_groups

    def get_vision_position_ids(self, start_position, grid_thw,
                                canvas_height=0, canvas_width=0,
                                h_offset=0, w_offset=0, spatial_merge_size=1):
        """3D (t,h,w) position ids for one grid of patches placed on a canvas."""
        llm_grid_h = int(grid_thw[0]) // spatial_merge_size
        llm_grid_w = int(grid_thw[1]) // spatial_merge_size
        canvas_height = canvas_height or llm_grid_h
        canvas_width = canvas_width or llm_grid_w
        h_coords = torch.linspace(0, canvas_height - 1, llm_grid_h).round().long() + h_offset
        w_coords = torch.linspace(0, canvas_width - 1, llm_grid_w).round().long() + w_offset
        H_grid, W_grid = torch.meshgrid(h_coords, w_coords, indexing="ij")
        T_grid = torch.zeros_like(H_grid)
        pos = torch.stack([T_grid, H_grid, W_grid], dim=0).reshape(3, -1)
        pos += start_position
        return pos

    def _get_rope_index_canvas(self, input_ids, mm_token_type_ids,
                               target_sizes, target_sizes_videos):
        """Canvas M-RoPE indices (3, B, S) plus rope_deltas (scalar)."""
        merge_factor = 2 if self.downsample_mode == "4x" else 4
        position_ids = torch.zeros(3, *input_ids.size(), dtype=torch.long)
        grid_iters = {
            1: iter(target_sizes.tolist()) if target_sizes is not None else None,
            2: iter(target_sizes_videos.tolist()) if target_sizes_videos is not None else None,
        }
        rope_deltas = 0
        for batch_idx in range(input_ids.shape[0]):
            cur_ids = input_ids[batch_idx].tolist()
            cur_mm = mm_token_type_ids[batch_idx].tolist()
            seq_len = len(cur_ids)
            curr_pos_ids = torch.arange(seq_len).expand(3, -1).clone()
            current_pos = 0
            current_idx = 0
            for group_start_idx, group_end_idx, frames in self._group_visual_frames(cur_ids, cur_mm):
                if group_start_idx > current_idx:
                    text_len = group_start_idx - current_idx
                    curr_pos_ids[:, current_idx:group_start_idx] = \
                        torch.arange(text_len) + current_pos
                    current_pos += text_len
                frame_cursor_idx = group_start_idx
                for modality, thumb_start_idx, thumb_end_idx, frame_end_idx, slices in frames:
                    target_sizes_thumb = next(grid_iters[modality])
                    frame_start_idx = thumb_start_idx - 1
                    if frame_start_idx > frame_cursor_idx:
                        gap_len = frame_start_idx - frame_cursor_idx
                        curr_pos_ids[:, frame_cursor_idx:frame_start_idx] = \
                            torch.arange(gap_len) + current_pos
                        current_pos += gap_len + 1
                    frame_end_idx = min(frame_end_idx, group_end_idx)
                    canvas_origin = current_pos
                    halo_before_canvas = max(canvas_origin - 1, 0)
                    llm_slice_h = llm_slice_w = 0
                    num_rows = num_cols = 0
                    canvas_height = target_sizes_thumb[0] // merge_factor
                    canvas_width = target_sizes_thumb[1] // merge_factor
                    target_sizes_first_slice = None
                    if slices:
                        num_cols = len(slices)
                        for k in range(len(slices) - 1):
                            if slices[k + 1][0] - slices[k][1] > 2:
                                num_cols = k + 1
                                break
                        num_rows = len(slices) // num_cols if num_cols > 0 else 1
                        if num_rows * num_cols != len(slices):
                            num_rows, num_cols = 1, len(slices)
                        target_sizes_first_slice = next(grid_iters[modality])
                        llm_slice_h = target_sizes_first_slice[0] // merge_factor
                        llm_slice_w = target_sizes_first_slice[1] // merge_factor
                        canvas_height = num_rows * llm_slice_h
                        canvas_width = num_cols * llm_slice_w
                    # base coat for whole frame, <image> -> halo just outside canvas
                    curr_pos_ids[:, frame_start_idx:frame_end_idx] = canvas_origin
                    curr_pos_ids[1:, frame_start_idx] = halo_before_canvas
                    # </image> closes the thumbnail on the far corner
                    if thumb_end_idx < frame_end_idx:
                        curr_pos_ids[1, thumb_end_idx] = canvas_origin + canvas_height
                        curr_pos_ids[2, thumb_end_idx] = canvas_origin + canvas_width
                    # thumbnail tokens (linspace-stretched over the full canvas)
                    curr_pos_ids[:, thumb_start_idx:thumb_end_idx] = \
                        self.get_vision_position_ids(canvas_origin, target_sizes_thumb,
                                                     canvas_height, canvas_width, 0, 0, merge_factor)
                    # slice tokens + <slice>/</slice> corner markers
                    for k, (slice_start_idx, slice_end_idx) in enumerate(slices):
                        h_off = (k // num_cols) * llm_slice_h
                        w_off = (k % num_cols) * llm_slice_w
                        target_sizes_slice = target_sizes_first_slice if k == 0 else next(grid_iters[modality])
                        slice_h = target_sizes_slice[0] // merge_factor
                        slice_w = target_sizes_slice[1] // merge_factor
                        ssm = slice_start_idx - 1
                        if ssm >= frame_start_idx:
                            curr_pos_ids[1, ssm] = canvas_origin + h_off
                            curr_pos_ids[2, ssm] = canvas_origin + w_off
                        if slice_end_idx < frame_end_idx:
                            curr_pos_ids[1, slice_end_idx] = canvas_origin + h_off + slice_h - 1
                            curr_pos_ids[2, slice_end_idx] = canvas_origin + w_off + slice_w - 1
                        curr_pos_ids[:, slice_start_idx:slice_end_idx] = \
                            self.get_vision_position_ids(canvas_origin, target_sizes_slice,
                                                         0, 0, h_off, w_off, merge_factor)
                    # "\n" ending a slice row sits just past the right edge of that row
                    for k in range(len(slices) - 1):
                        gap_start_idx, gap_end_idx = slices[k][1], slices[k + 1][0]
                        if gap_end_idx - gap_start_idx <= 2:
                            continue
                        boundary_h = (k // num_cols + 1) * llm_slice_h - 1
                        right_edge_w = num_cols * llm_slice_w
                        for newline_idx in range(gap_start_idx + 1, gap_end_idx - 1):
                            curr_pos_ids[1, newline_idx] = canvas_origin + boundary_h
                            curr_pos_ids[2, newline_idx] = canvas_origin + right_edge_w
                    current_pos = canvas_origin + max(canvas_height, canvas_width) + 1
                    frame_cursor_idx = frame_end_idx
                if frame_cursor_idx < group_end_idx:
                    trail_len = group_end_idx - frame_cursor_idx
                    curr_pos_ids[:, frame_cursor_idx:group_end_idx] = \
                        torch.arange(trail_len) + current_pos
                    current_pos += trail_len
                current_idx = group_end_idx
            if current_idx < seq_len:
                curr_pos_ids[:, current_idx:seq_len] = \
                    torch.arange(seq_len - current_idx) + current_pos
            position_ids[:, batch_idx, :] = curr_pos_ids
        # rope_deltas = max position + 1 - seq_len  (matches HF: amax over all
        # channels/batches, not just the per-batch current_pos)
        rope_deltas = int(position_ids.amax(dim=(0, 2)).max().item()) + 1 - input_ids.shape[1]
        return position_ids, rope_deltas

    # ======================== Forward ========================

    def forward_prefill(self, position_ids):
        if self.model.history_length == 0 or not self.support_history:
            self.history_max_posid = 0
            return self.model.forward_first(position_ids)
        self.max_posid += self.history_max_posid
        position_ids = position_ids + self.history_max_posid
        return self.model.forward_first(position_ids)

    # ======================== Inference ========================

    def run_once(self, input_str, media_path=""):
        """
        Run a single inference turn programmatically.

        Returns the generated text (stdout streaming is preserved), or
        None if the input could not be processed.
        """
        media_path = (media_path or "").strip()
        if media_path == "":
            media_type = "text"
        elif not os.path.exists(media_path):
            print("Can't find image or video: {}".format(media_path))
            return None
        else:
            media_type = self.get_media_type(media_path)

        inputs = self.process(input_str, media_type, media_path)
        token_len = inputs.input_ids.numel()
        max_input_tokens = self.model.SEQLEN if self.model.support_history \
            else self.model.MAX_INPUT_LENGTH
        if token_len > max_input_tokens:
            ts_key = "target_sizes_videos" if media_type == "video" else "target_sizes"
            if hasattr(inputs, ts_key):
                print("target_sizes:{}".format(getattr(inputs, ts_key)))
            print(
                "Error: The maximum question length should be shorter than {} but we get {} instead."
                .format(max_input_tokens, token_len))
            return None
        if self.support_history:
            if (token_len + self.model.history_length > self.model.SEQLEN - 128) or \
            (self.model.history_length > self.model.PREFILL_KV_LENGTH):
                print("Warning: History is full and clear it to continue.")
                self.model.clear_history()
                self.history_max_posid = 0
        print("\nAnswer:")

        # Chat
        first_start = time.time()
        self.model.forward_embed(inputs.input_ids.numpy())
        vit_start = vit_end = 0
        vision_target_sizes = None
        rope_deltas = 0
        if media_type in ("image", "video"):
            vit_start = time.time()
            if media_type == "image":
                self.vit_process_image(inputs)
                pad_id = self.ID_IMAGE_PAD
                target_sizes = inputs.target_sizes
                target_sizes_videos = None
                vision_target_sizes = inputs.target_sizes
            else:
                self.vit_process_video(inputs)
                pad_id = self.ID_VIDEO_PAD
                target_sizes = None
                target_sizes_videos = inputs.target_sizes_videos
                vision_target_sizes = inputs.target_sizes_videos
            vit_end = time.time()
            if self.is_canvas:
                position_ids, rope_deltas = self.get_rope_index(
                    inputs.input_ids, mm_token_type_ids=inputs.mm_token_type_ids,
                    target_sizes=target_sizes, target_sizes_videos=target_sizes_videos)
            else:
                position_ids, rope_deltas = self.get_rope_index(
                    inputs.input_ids, pad_id=pad_id, target_sizes=target_sizes)
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
            # decode: 3-channel position = text_pos + rope_deltas
            position_ids = np.array(
                [self.max_posid + rope_deltas] * 3, dtype=np.int32)
            token = self.model.forward_next(position_ids)
            tok_num += 1
        self.history_max_posid = self.max_posid + 2
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration if next_duration > 0 else 0.0
        print(f"\nFTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} tokens/s")
        if vision_target_sizes is not None:
            print(f"Vision({vision_target_sizes.tolist()}): {vit_end - vit_start:.3f} s")
        if self.support_history:
            print(f"Total Tokens: {self.model.history_length}")
        return text

    def chat(self):
        """
        Start an interactive chat session.
        """
        # Instruct
        print("""\n=================================================================
1. If you want to quit, please enter one of [/q, /quit, /exit]
2. To create a new chat session, please enter one of [/clear, /new]
3. To ask about an image or video, include @<path> in your question
4. To use the contents of a .txt or .md file as your question, include @<path>
=================================================================""")
        # Stop Chatting with "/exit" input
        while True:
            input_str = input("\nQuestion: ")
            # Quit
            if input_str in ["/exit", "/q", "/quit"]:
                break
            if input_str in ["/clear", "/new", "/c"]:
                print("New chat session created.")
                self.model.clear_history()
                self.history_max_posid = 0
                continue

            # Media files are attached with @path in the question
            input_str, media_path = extract_media(input_str)
            self.run_once(input_str, media_path)


def read_prompt_file(path):
    """Read a @-referenced .txt/.md file and return its contents."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except OSError as e:
        print(f"Cannot open prompt file [ {path} ]: {e}")
        sys.exit(1)
    # Trim trailing newlines so the file behaves like a typed prompt.
    return content.rstrip('\r\n')


def extract_media(input_str):
    """Split @<path> attachments out of the input text.

    An @<path> ending in .txt or .md is read as prompt text and replaces the
    token inline; any other @<path> is treated as an image/video attachment.
    """
    media_paths = []
    text_tokens = []
    for t in input_str.split():
        if t.startswith("@") and len(t) > 1:
            path = t[1:]
            if path.lower().endswith((".txt", ".md")):
                text_tokens.append(read_prompt_file(path))
            else:
                media_paths.append(path)
        else:
            text_tokens.append(t)
    input_str = " ".join(text_tokens)
    if len(media_paths) > 1:
        print("Only one media file is supported, using: {}".format(media_paths[0]))
    media_path = media_paths[0] if media_paths else ""
    return input_str, media_path


def main(args):
    model = MiniCPMV(args)
    if args.prompt is not None:
        # Programmatic (non-interactive) mode: run once and exit.
        prompt, media_path = extract_media(args.prompt)
        model.run_once(prompt, media_path)
    else:
        model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="../config",
                        help='path to the processor file')
    parser.add_argument('--downsample_mode', type=str, default='16x',
                        choices=['4x', '16x'], help='ViT downsample mode')
    parser.add_argument('--max_num_frames', type=int, default=16,
                        help='Maximum number of video frames to sample')
    parser.add_argument('--max_slice_nums', type=int, default=None,
                        help='Maximum number of slices (default: 36 for image, 1 for video)')
    parser.add_argument('-d', '--devid', type=int, default=0, help='device ID to use')
    parser.add_argument('-p', '--prompt', type=str, default=None,
                        help='If set, run programmatically (non-interactive): a single inference is performed using this prompt and then the program exits. Include @<path> to attach an image or video, or to read prompt text from a .txt/.md file.')
    # yapf: enable
    args = parser.parse_args()
    main(args)
