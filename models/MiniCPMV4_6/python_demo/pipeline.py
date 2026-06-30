# ==============================================================================
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import time
import argparse
import warnings
from transformers import AutoProcessor
import chat
import os
import sys
import torch
import numpy as np

warnings.filterwarnings("ignore", message=".*torchcodec.*")


class MiniCPMV4_6():

    def __init__(self, args):
        # devid
        self.device = args.devid

        # load model
        self.model = chat.MiniCPMV4_6()
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
                    max_slice_nums=self.max_slice_nums or 36))
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
        """MiniCPM-V-4.6 bucketize position encoding."""
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

    def get_rope_index(self, input_ids: torch.LongTensor, target_sizes: torch.LongTensor,
                       pad_id: int) -> torch.Tensor:
        """Compute 3D MRoPE position IDs for the text model.

        MiniCPM token structure per image:
            <image> pad×N </image>          ← main block → target_sizes[i]
            <slice> pad×M </slice>          ← slice 0    → target_sizes[i+1]
            ... (may have \\n between slice rows)
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
            vision_start_indices = torch.argwhere(input_ids == self.ID_IMAGE_TOKEN).squeeze(1)
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
                    if input_tokens[k] == self.ID_SLICE_START:
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
                pad_id = self.ID_IMAGE_PAD
                vision_target_sizes = inputs.target_sizes
            else:
                self.vit_process_video(inputs)
                pad_id = self.ID_VIDEO_PAD
                vision_target_sizes = inputs.target_sizes_videos
            vit_end = time.time()
            position_ids = self.get_rope_index(inputs.input_ids, vision_target_sizes, pad_id)
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
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
=================================================================""")
        # Stop Chatting with "exit" input
        while True:
            input_str = input("\nQuestion: ")
            # Quit
            if input_str in ["exit", "q", "quit"]:
                break
            if input_str in ["clear", "new", "c"]:
                print("New chat session created.")
                self.model.clear_history()
                self.history_max_posid = 0
                continue

            media_path = input("\nImage or Video Path: ")
            self.run_once(input_str, media_path)


def main(args):
    model = MiniCPMV4_6(args)
    if args.prompt is not None:
        # Programmatic (non-interactive) mode: run once and exit.
        model.run_once(args.prompt, args.media_path)
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
                        help='If set, run programmatically (non-interactive): a single inference is performed using this prompt and then the program exits.')
    parser.add_argument('-t', '--prompt_file', type=str, default=None,
                        help='Path to a text file whose contents are used as the programmatic mode prompt. If --prompt is also set, the file contents come first, followed by the --prompt value (combined with a newline).')
    parser.add_argument('--media_path', type=str, default="",
                        help='Path to an image or video for programmatic mode (used together with --prompt). Leave empty for text-only.')
    # yapf: enable
    args = parser.parse_args()
    # If --prompt_file was provided, load the prompt text from that file.
    if args.prompt_file is not None:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                file_prompt = f.read()
        except OSError as e:
            print(f"Cannot open prompt file [ {args.prompt_file} ]: {e}")
            sys.exit(1)
        file_prompt = file_prompt.rstrip('\r\n')
        if args.prompt is None or args.prompt == "":
            args.prompt = file_prompt
        else:
            args.prompt = file_prompt + "\n" + args.prompt
    main(args)
