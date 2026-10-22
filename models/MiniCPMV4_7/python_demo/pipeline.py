# ==============================================================================
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import time
import argparse
import itertools
import warnings
from transformers import AutoProcessor
import chat
import os
import sys
import torch
import numpy as np

warnings.filterwarnings("ignore", message=".*torchcodec.*")


class MiniCPMV4_7():

    def __init__(self, args):
        # devid
        self.device = args.devid

        # load model
        self.model = chat.MiniCPMV4_7()
        self.model.init(self.device, args.model_path)
        self.processor = AutoProcessor.from_pretrained(args.config_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        # Special token IDs
        self.ID_IMAGE_TOKEN = self.tokenizer.convert_tokens_to_ids('<image>')  # 248078
        self.ID_IMAGE_PAD = self.tokenizer.convert_tokens_to_ids('<|image_pad|>')  # 248056
        self.ID_IMAGE_END = self.tokenizer.convert_tokens_to_ids('</image>')  # 248079
        self.ID_VIDEO_TOKEN = self.tokenizer.convert_tokens_to_ids('<video>')  # 248077
        self.ID_VIDEO_PAD = self.tokenizer.convert_tokens_to_ids('<|video_pad|>')  # 248057
        self.ID_SLICE_START = self.tokenizer.convert_tokens_to_ids('<slice>')  # 248088
        self.ID_SLICE_END = self.tokenizer.convert_tokens_to_ids('</slice>')  # 248089
        self.ID_NEWLINE = self.tokenizer.encode('\n', add_special_tokens=False)[0]  # 198
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids('<|im_end|>')

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
                    max_slice_nums=self.max_slice_nums or 9,
                    max_num_frames=self.max_num_frames,
                    stack_frames=1, use_image_id=False))
        else:  # video
            return self.processor.apply_chat_template(
                messages, **common,
                processor_kwargs=dict(
                    downsample_mode=self.downsample_mode,
                    max_slice_nums=self.max_slice_nums or 1,
                    max_num_frames=self.max_num_frames,
                    stack_frames=1, use_image_id=False))

    # ======================== Index computation ========================

    def compute_pos_ids(self, h_patches, w_patches):
        """MiniCPM-V-4.7 bucketize position encoding."""
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
            if tid == self.ID_IMAGE_TOKEN or tid == self.ID_SLICE_START:
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
        block_positions = [i for i, tid in enumerate(ids) if tid == self.ID_IMAGE_TOKEN]
        self._vit_process(inputs.pixel_values_videos, inputs.target_sizes_videos,
                          block_positions)

    # ======================== Position IDs for text model ========================

    @staticmethod
    def _frame_end_idx(last_crop_end_idx, input_tokens, markers, limit):
        """End of a frame's span, including the markers that close it."""
        end_idx = last_crop_end_idx
        while end_idx < limit and input_tokens[end_idx] in markers:
            end_idx += 1
        return end_idx

    def _group_visual_frames(self, input_tokens, mm_token_type_ids):
        """Split one sequence into visual groups (port of the HF reference).

        A slice is one detail crop wrapped in <slice>/</slice>, a frame is one
        picture/video frame (thumbnail plus its slices), and a group is a run of
        frames separated only by markers/newlines, which therefore share one
        canvas budget.  Returns (group_start_idx, group_end_idx, frames) per
        group, where frames holds
        (modality, thumb_start_idx, thumb_end_idx, frame_end_idx, slices).
        """
        markers = (
            self.ID_SLICE_START,
            self.ID_SLICE_END,
            self.ID_IMAGE_TOKEN,
            self.ID_IMAGE_END,
            self.ID_NEWLINE,
        )
        slice_start_id = markers[0]

        seq_len = len(input_tokens)
        groups = []
        open_slices = None
        previous_end_idx = 0
        for modality, run in itertools.groupby(enumerate(mm_token_type_ids),
                                               lambda item: item[1]):
            if modality == 0:
                continue
            run = list(run)
            crop = (run[0][0], run[-1][0] + 1)

            if open_slices is not None and input_tokens[crop[0] - 1] == slice_start_id:
                open_slices.append(crop)
            else:
                open_slices = []
                frame = (modality, crop[0], crop[1], open_slices)
                gap = input_tokens[previous_end_idx:crop[0] - 1]
                adjacent = bool(groups) and all(tid in markers for tid in gap)
                if adjacent:
                    groups[-1][1].append(frame)
                else:
                    groups.append((crop[0] - 1, [frame]))
            previous_end_idx = crop[1]

        visual_groups = []
        for group_start_idx, frames in groups:
            frames = [
                (
                    modality,
                    thumb_start_idx,
                    thumb_end_idx,
                    self._frame_end_idx(
                        slices[-1][1] if slices else thumb_end_idx,
                        input_tokens, markers, seq_len),
                    slices,
                )
                for modality, thumb_start_idx, thumb_end_idx, slices in frames
            ]
            group_end_idx = frames[-1][3]
            visual_groups.append((group_start_idx, group_end_idx, frames))
        return visual_groups

    @staticmethod
    def get_vision_position_ids(start_position, grid_h, grid_w, canvas_height=0,
                                canvas_width=0, h_offset=0, w_offset=0,
                                spatial_merge_size=1):
        """3D (t, h, w) position ids for one patch grid placed on a canvas.

        canvas_height/canvas_width == 0 means "no canvas given": the canvas then
        defaults to the grid's own size, so the linspace resample degenerates to
        a plain arange.  With a canvas, the grid is stretched over it.
        """
        llm_grid_h = grid_h // spatial_merge_size
        llm_grid_w = grid_w // spatial_merge_size
        canvas_height = canvas_height or llm_grid_h
        canvas_width = canvas_width or llm_grid_w
        h_coords = torch.linspace(0, canvas_height - 1, llm_grid_h).round().long() + h_offset
        w_coords = torch.linspace(0, canvas_width - 1, llm_grid_w).round().long() + w_offset
        H_grid, W_grid = torch.meshgrid(h_coords, w_coords, indexing="ij")
        T_grid = torch.zeros_like(H_grid)
        position_ids = torch.stack([T_grid, H_grid, W_grid], dim=0).reshape(3, -1)
        return position_ids + start_position

    def get_rope_index(self, input_ids, mm_token_type_ids,
                       target_sizes=None, target_sizes_videos=None):
        """Canvas M-RoPE position ids (3, batch, seq) for MiniCPM-V-4.7.

        Port of MiniCPMV4_7Model.get_rope_index (transformers): each frame owns
        a canvas -- the thumbnail is stretched over it via linspace, slice
        markers pin their corners, <image> gets a halo position, newlines sit on
        slice-row boundaries, and every text span keeps plain 1-D positions.
        The compiled bmodel gathers its rope table with these position ids, so
        the bmodel itself is agnostic to the scheme.
        """
        merge_factor = self.spatial_merge_size

        position_ids = torch.zeros(3, *input_ids.size(), dtype=torch.long)
        grid_iters = {
            1: iter(target_sizes.tolist()) if target_sizes is not None else None,
            2: iter(target_sizes_videos.tolist()) if target_sizes_videos is not None else None,
        }

        for batch_idx in range(input_ids.shape[0]):
            current_input_ids = input_ids[batch_idx].tolist()
            current_mm_ids = mm_token_type_ids[batch_idx].tolist()
            seq_len = len(current_input_ids)
            # Default coat is plain 1-D positions, overwritten per span below.
            curr_position_ids = torch.arange(seq_len).expand(3, -1).clone()

            current_pos = 0
            current_idx = 0
            for group_start_idx, group_end_idx, frames in self._group_visual_frames(
                    current_input_ids, current_mm_ids):
                # Text in front of the group is plain 1-D.
                if group_start_idx > current_idx:
                    text_len = group_start_idx - current_idx
                    curr_position_ids[:, current_idx:group_start_idx] = \
                        torch.arange(text_len) + current_pos
                    current_pos += text_len

                frame_cursor_idx = group_start_idx
                for modality, thumb_start_idx, thumb_end_idx, frame_end_idx, slices in frames:
                    target_sizes_thumb = next(grid_iters[modality])
                    frame_start_idx = thumb_start_idx - 1

                    # Tokens between two frames of one clip are 1-D text; the
                    # extra step afterwards keeps the next frame's halo off the
                    # last of them.
                    if frame_start_idx > frame_cursor_idx:
                        gap_len = frame_start_idx - frame_cursor_idx
                        curr_position_ids[:, frame_cursor_idx:frame_start_idx] = \
                            torch.arange(gap_len) + current_pos
                        current_pos += gap_len + 1

                    frame_end_idx = min(frame_end_idx, group_end_idx)
                    canvas_origin = current_pos
                    halo_before_canvas = max(canvas_origin - 1, 0)

                    llm_slice_h, llm_slice_w = 0, 0
                    num_rows, num_cols = 0, 0
                    canvas_height = target_sizes_thumb[0] // merge_factor
                    canvas_width = target_sizes_thumb[1] // merge_factor
                    if slices:
                        # Slices are laid out row-major; a gap wider than the two
                        # </slice><slice> markers is the newline ending a row.
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

                    # Base coat for the whole frame, then <image> -> halo just
                    # outside the canvas.
                    curr_position_ids[:, frame_start_idx:frame_end_idx] = canvas_origin
                    curr_position_ids[1:, frame_start_idx] = halo_before_canvas

                    # </image> closes the thumbnail on the far corner of the canvas.
                    if thumb_end_idx < frame_end_idx:
                        curr_position_ids[1, thumb_end_idx] = canvas_origin + canvas_height
                        curr_position_ids[2, thumb_end_idx] = canvas_origin + canvas_width

                    # Thumbnail tokens, stretched over the full canvas when
                    # slices follow (linspace degenerates to arange without them).
                    curr_position_ids[:, thumb_start_idx:thumb_end_idx] = \
                        self.get_vision_position_ids(
                            start_position=canvas_origin,
                            grid_h=target_sizes_thumb[0],
                            grid_w=target_sizes_thumb[1],
                            canvas_height=canvas_height,
                            canvas_width=canvas_width,
                            spatial_merge_size=merge_factor)

                    # Slice tokens plus the <slice>/</slice> markers pinning
                    # corners.
                    for k, (slice_start_idx, slice_end_idx) in enumerate(slices):
                        h_off = (k // num_cols) * llm_slice_h
                        w_off = (k % num_cols) * llm_slice_w

                        target_sizes_slice = target_sizes_first_slice if k == 0 \
                            else next(grid_iters[modality])
                        slice_h = target_sizes_slice[0] // merge_factor
                        slice_w = target_sizes_slice[1] // merge_factor

                        slice_start_marker_idx = slice_start_idx - 1
                        if slice_start_marker_idx >= frame_start_idx:
                            curr_position_ids[1, slice_start_marker_idx] = \
                                canvas_origin + h_off
                            curr_position_ids[2, slice_start_marker_idx] = \
                                canvas_origin + w_off
                        if slice_end_idx < frame_end_idx:
                            curr_position_ids[1, slice_end_idx] = \
                                canvas_origin + h_off + slice_h - 1
                            curr_position_ids[2, slice_end_idx] = \
                                canvas_origin + w_off + slice_w - 1

                        curr_position_ids[:, slice_start_idx:slice_end_idx] = \
                            self.get_vision_position_ids(
                                start_position=canvas_origin,
                                grid_h=target_sizes_slice[0],
                                grid_w=target_sizes_slice[1],
                                h_offset=h_off,
                                w_offset=w_off,
                                spatial_merge_size=merge_factor)

                    # The "\n" that ends a row of slices sits just past the
                    # right edge of that row.
                    for k in range(len(slices) - 1):
                        gap_start_idx, gap_end_idx = slices[k][1], slices[k + 1][0]
                        if gap_end_idx - gap_start_idx <= 2:
                            continue
                        boundary_h = (k // num_cols + 1) * llm_slice_h - 1
                        right_edge_w = num_cols * llm_slice_w
                        for newline_idx in range(gap_start_idx + 1, gap_end_idx - 1):
                            curr_position_ids[1, newline_idx] = canvas_origin + boundary_h
                            curr_position_ids[2, newline_idx] = canvas_origin + right_edge_w

                    current_pos = canvas_origin + max(canvas_height, canvas_width) + 1
                    frame_cursor_idx = frame_end_idx

                if frame_cursor_idx < group_end_idx:
                    trail_len = group_end_idx - frame_cursor_idx
                    curr_position_ids[:, frame_cursor_idx:group_end_idx] = \
                        torch.arange(trail_len) + current_pos
                    current_pos += trail_len
                current_idx = group_end_idx

            if current_idx < seq_len:
                curr_position_ids[:, current_idx:seq_len] = \
                    torch.arange(seq_len - current_idx) + current_pos
            position_ids[:, batch_idx, :] = curr_position_ids
        return position_ids.to(torch.int32)

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
        if media_type in ("image", "video"):
            vit_start = time.time()
            if media_type == "image":
                self.vit_process_image(inputs)
                vision_target_sizes = inputs.target_sizes
            else:
                self.vit_process_video(inputs)
                vision_target_sizes = inputs.target_sizes_videos
            vit_end = time.time()
            if media_type == "image":
                position_ids = self.get_rope_index(
                    inputs.input_ids, inputs.mm_token_type_ids,
                    target_sizes=inputs.target_sizes)
            else:
                position_ids = self.get_rope_index(
                    inputs.input_ids, inputs.mm_token_type_ids,
                    target_sizes_videos=inputs.target_sizes_videos)
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
            if "\ufffd" not in word:
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
    model = MiniCPMV4_7(args)
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
