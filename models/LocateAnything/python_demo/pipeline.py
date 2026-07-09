# ==============================================================================
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import time
import argparse
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import chat


class LocateAnything():

    def __init__(self, args):
        self.device = args.devid

        # load bmodel
        self.model = chat.LocateAnything()
        self.model.init(self.device, args.model_path)

        # load processor from model config (trust_remote_code needed for custom processor)
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            args.config_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        # special token IDs
        self.ID_IM_END = self.tokenizer.eos_token_id or 151645
        self.ID_IMAGE_TOKEN = 151665  # image_token_index from config
        # bounding box tokens for output parsing
        self.BOX_START = 151668
        self.BOX_END = 151669
        self.COORD_START = 151677
        self.COORD_END = 152677
        self.NONE_TOKEN = 4064

        self.patch_size = 14
        self.merge_kernel_size = [2, 2]
        self.spatial_merge_size = 2
        self.embed_dim = 1152
        self.head_dim = 72
        self.support_history = self.model.support_history
        self.max_posid = 0
        self.history_max_posid = 0

        # load pos_emb weight for interpolation
        self.pos_emb_weight = None

    def compute_merger_index(self, grid_hw):
        """Compute 2×2 spatial merge reorder index.

        Args:
            grid_hw: tuple (h, w) — grid height and width in patches

        Returns:
            numpy array [h*w], int32 — reorder index for GatherOp
        """
        h, w = grid_hw
        new_h = h // self.merge_kernel_size[0]
        new_w = w // self.merge_kernel_size[1]
        indices = np.zeros(new_h * new_w * 4, dtype=np.int32)
        for i in range(new_h):
            for j in range(new_w):
                out_pos = i * new_w + j
                base = out_pos * 4
                indices[base] = (2 * i) * w + 2 * j
                indices[base + 1] = (2 * i) * w + 2 * j + 1
                indices[base + 2] = (2 * i + 1) * w + 2 * j
                indices[base + 3] = (2 * i + 1) * w + 2 * j + 1
        return indices

    def load_pos_emb_weight(self, config_path):
        """Load the learnable 2D position embedding from exported npz."""
        if self.pos_emb_weight is not None:
            return
        pos_emb_file = os.path.join(config_path, "vit_pos_emb.npz")
        if os.path.exists(pos_emb_file):
            data = np.load(pos_emb_file)
            self.pos_emb_weight = torch.from_numpy(data['pos_emb'])
        else:
            raise FileNotFoundError(
                f"vit_pos_emb.npz not found in {config_path}. "
                "Recompile the bmodel to generate it.")

    def compute_pos_emb(self, grid_hw, config_path):
        """Compute interpolated position embedding for given grid size."""
        self.load_pos_emb_weight(config_path)
        h, w = grid_hw
        pos = self.pos_emb_weight
        if (h, w) != (pos.shape[0], pos.shape[1]):
            pos = F.interpolate(
                pos.permute(2, 0, 1).unsqueeze(0),
                size=(h, w),
                mode='bicubic',
            ).squeeze(0).permute(1, 2, 0)
        return pos.flatten(end_dim=1).numpy().astype(np.float32)

    def compute_rope(self, grid_hw):
        """Compute 2D RoPE cos/sin for given grid size.

        Returns interleaved layout: [w_cos0, h_cos0, w_cos1, h_cos1, ...]
        matching HF's Rope2DPosEmb._precompute_freqs_cis.
        """
        h, w = grid_hw
        dim = self.head_dim  # 72
        quarter = dim // 4  # 18
        half_dim = dim // 2  # 36
        theta_base = 10000.0

        dim_range = np.arange(0, dim, 4, dtype=np.float32)[:quarter]
        freqs = 1.0 / (theta_base ** (dim_range / dim))

        # H and W direction
        h_pos = np.arange(h, dtype=np.float32)
        h_freqs = np.outer(h_pos, freqs)
        h_cos = np.cos(h_freqs)
        h_sin = np.sin(h_freqs)

        w_pos = np.arange(w, dtype=np.float32)
        w_freqs = np.outer(w_pos, freqs)
        w_cos = np.cos(w_freqs)
        w_sin = np.sin(w_freqs)

        # Build interleaved: [w_cos0, h_cos0, w_cos1, h_cos1, ...]
        cos = np.zeros((h, w, half_dim), dtype=np.float32)
        sin = np.zeros((h, w, half_dim), dtype=np.float32)
        for i in range(quarter):
            cos[:, :, 2 * i] = w_cos[None, :, i]
            sin[:, :, 2 * i] = w_sin[None, :, i]
            cos[:, :, 2 * i + 1] = h_cos[:, i:i + 1]
            sin[:, :, 2 * i + 1] = h_sin[:, i:i + 1]

        cos = cos.reshape(-1, half_dim)
        sin = sin.reshape(-1, half_dim)
        return cos, sin

    def text_message(self):
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": self.input_str}],
        }]
        return messages

    def image_message(self, path):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": path},
                {"type": "text", "text": self.input_str},
            ],
        }]
        return messages

    def get_media_type(self, file_path):
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext in image_exts:
            return "image"
        raise RuntimeError(f"Unsupported media type: {ext}")

    def process(self, messages):
        """Process messages using LocateAnything's custom processor."""
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)

        if images is not None:
            # Process images through the custom image processor
            image_inputs = self.processor.image_processor(images=images)
            pixel_values = image_inputs['pixel_values']
            image_grid_hws = image_inputs['image_grid_hws']

            # Expand <image-N> placeholders to <img><IMG_CONTEXT>*num_tokens</img>
            merge_ratio = self.merge_kernel_size[0] * self.merge_kernel_size[1]
            image_token = self.processor.image_token  # <IMG_CONTEXT>
            image_start = self.processor.image_start_token  # <img>
            image_end = self.processor.image_end_token  # </img>
            for idx, grid_hw in enumerate(image_grid_hws):
                h, w = int(grid_hw[0]), int(grid_hw[1])
                num_tokens = h * w // merge_ratio
                placeholder = f"<image-{idx + 1}>"
                expansion = f"{image_start}{image_token * num_tokens}{image_end}"
                text = text.replace(placeholder, expansion, 1)
        else:
            pixel_values = None
            image_grid_hws = None

        # Tokenize the text
        text_inputs = self.processor.tokenizer(
            [text], padding=False, return_tensors="pt")

        return {
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs.get('attention_mask'),
            'pixel_values': pixel_values,
            'image_grid_hws': image_grid_hws,
        }

    def vit_process_image(self, inputs, vit_offset):
        """Process image through ViT bmodel."""
        pixel_values = inputs['pixel_values']  # [N, 3, 14, 14]
        image_grid_hws = inputs['image_grid_hws']  # [[h, w]]

        # Flatten patches: [N, 3, 14, 14] -> [N, 588]
        N = pixel_values.shape[0]
        pixel_flat = pixel_values.reshape(N, -1).numpy().astype(np.float32)

        # Process each image (usually just one)
        offset = 0
        for grid_hw in image_grid_hws:
            h, w = int(grid_hw[0]), int(grid_hw[1])
            num_patches = h * w
            # Compute runtime inputs for this image
            merger_idx = self.compute_merger_index((h, w))
            pos_emb = self.compute_pos_emb((h, w), self._config_path)
            rope_cos, rope_sin = self.compute_rope((h, w))
            # Extract patches for this image
            patches = pixel_flat[offset:offset + num_patches]
            # Call ViT bmodel with 5 inputs
            self.model.forward_vit(patches, merger_idx, pos_emb,
                                   rope_cos, rope_sin, vit_offset)
            # Update offsets
            merged_tokens = num_patches // (self.spatial_merge_size**2)
            vit_offset += merged_tokens
            offset += num_patches

    # Regex for <box> blocks: 4 coords = box, 2 coords = point.
    # The point regex requires </box> right after the 2nd coord, so it does
    # not match a 4-coord box.
    _BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
    _POINT_RE = re.compile(r"<box><(\d+)><(\d+)></box>")
    _REF_RE = re.compile(r"<ref>(.*?)</ref>")

    @staticmethod
    def parse_boxes(answer, image_width, image_height):
        """Parse 4-coord <box> blocks into pixel-coordinate bounding boxes.

        Coordinates in model output are normalized integers in [0, 1000].
        """
        boxes = []
        for m in LocateAnything._BOX_RE.finditer(answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append({
                "x1": x1 / 1000 * image_width,
                "y1": y1 / 1000 * image_height,
                "x2": x2 / 1000 * image_width,
                "y2": y2 / 1000 * image_height,
            })
        return boxes

    @staticmethod
    def parse_points(answer, image_width, image_height):
        """Parse 2-coord <box> blocks into pixel-coordinate points."""
        points = []
        for m in LocateAnything._POINT_RE.finditer(answer):
            x, y = int(m.group(1)), int(m.group(2))
            points.append({
                "x": x / 1000 * image_width,
                "y": y / 1000 * image_height,
            })
        return points

    @staticmethod
    def parse_result(answer, image_width, image_height):
        """Parse output preserving <ref> -> boxes/points association.

        Returns a list of {"ref": name, "boxes": [...], "points": [...]}.
        Boxes/points that precede any <ref> are grouped under ref "".
        """
        results = []
        refs = list(LocateAnything._REF_RE.finditer(answer))
        # segment before the first <ref> (usually empty)
        first_start = refs[0].start() if refs else len(answer)
        if first_start > 0:
            head = answer[:first_start]
            boxes = LocateAnything.parse_boxes(head, image_width,
                                                image_height)
            points = LocateAnything.parse_points(head, image_width,
                                                  image_height)
            if boxes or points:
                results.append({"ref": "", "boxes": boxes, "points": points})
        for i, rm in enumerate(refs):
            ref_name = rm.group(1)
            start = rm.end()
            end = refs[i + 1].start() if i + 1 < len(refs) else len(answer)
            segment = answer[start:end]
            boxes = LocateAnything.parse_boxes(segment, image_width,
                                                image_height)
            points = LocateAnything.parse_points(segment, image_width,
                                                  image_height)
            if boxes or points:
                results.append({
                    "ref": ref_name, "boxes": boxes, "points": points
                })
        return results


    def get_rope_index(self, input_ids, num_image_tokens=0):
        """Compute 1D position_ids for standard Qwen2 RoPE.

        Args:
            input_ids: [1, seq_len] tensor
            num_image_tokens: number of image tokens in the sequence

        Returns:
            numpy array [seq_len], int32
        """
        seq_len = input_ids.shape[1]
        position_ids = np.arange(seq_len, dtype=np.int32)
        return position_ids

    def forward_prefill(self, position_ids):
        if self.model.history_length == 0 or not self.support_history:
            self.history_max_posid = 0
            return self.model.forward_first(position_ids)
        self.max_posid += self.history_max_posid
        position_ids = position_ids + self.history_max_posid
        return self.model.forward_first(position_ids)

    def run_once(self, input_str, media_path=""):
        self.input_str = input_str
        self.config_path = getattr(self, '_config_path', '')
        media_path = (media_path or "").strip()
        if media_path == "":
            messages = self.text_message()
            media_type = "text"
        elif not os.path.exists(media_path):
            print("Can't find image: {}".format(media_path))
            return None
        else:
            media_type = self.get_media_type(media_path)
            if media_type == "image":
                messages = self.image_message(media_path)
            else:
                print("Unsupported media type: {}".format(media_path))
                return None

        inputs = self.process(messages)
        input_ids = inputs['input_ids']
        token_len = input_ids.numel()
        if token_len > self.model.MAX_INPUT_LENGTH:
            print(
                "Error: The maximum question length should be shorter than {}"
                " but we get {} instead.".format(
                    self.model.MAX_INPUT_LENGTH, token_len))
            return None

        if self.support_history:
            if (token_len + self.model.history_length > self.model.SEQLEN - 128) or \
               (self.model.history_length > self.model.PREFILL_KV_LENGTH):
                print("Warning: History is full and clear it to continue.")
                self.model.clear_history()
                self.history_max_posid = 0

        print("\nAnswer:")
        first_start = time.time()

        # Embed text tokens
        self.model.forward_embed(input_ids.squeeze(0).tolist())

        # Process image if present
        vit_start = vit_end = 0
        if media_type == "image" and inputs['pixel_values'] is not None:
            # Find where image tokens are in the sequence
            vit_token_list = torch.where(
                input_ids == self.ID_IMAGE_TOKEN)[1].tolist()
            vit_offset = vit_token_list[0] if vit_token_list else 0
            vit_start = time.time()
            self.vit_process_image(inputs, vit_offset)
            vit_end = time.time()

        # Compute position_ids (1D for standard Qwen2 RoPE)
        position_ids = self.get_rope_index(input_ids)
        self.max_posid = int(position_ids.max())
        token = self.forward_prefill(position_ids)

        first_end = time.time()
        tok_num = 0

        # Generate following tokens
        full_word_tokens = []
        text = ""
        while token not in [self.ID_IM_END] and \
              self.model.history_length < self.model.SEQLEN:
            full_word_tokens.append(token)
            word = self.tokenizer.decode(
                full_word_tokens, skip_special_tokens=False)
            if "�" not in word:
                if len(full_word_tokens) == 1:
                    pre_word = word
                    word = self.tokenizer.decode(
                        [token, token],
                        skip_special_tokens=False)[len(pre_word):]
                text += word
                print(word, flush=True, end="")
                full_word_tokens = []
            self.max_posid += 1
            position_ids = np.array([self.max_posid], dtype=np.int32)
            token = self.model.forward_next(position_ids)
            tok_num += 1

        self.history_max_posid = self.max_posid + 2
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration if next_duration > 0 else 0.0
        print(f"\nFTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")
        if media_type == "image" and inputs.get('image_grid_hws') is not None:
            print(f"VIT({inputs['image_grid_hws'].tolist()}): "
                  f"{vit_end - vit_start:.3f} s")
            # parse normalized [0,1000] coords -> pixel coordinates
            with Image.open(media_path) as img:
                w, h = img.size
            results = self.parse_result(text, w, h)
            if results:
                print("Parsed (pixel coords):")
                for r in results:
                    label = r["ref"] or "(no ref)"
                    for b in r["boxes"]:
                        print(f"  [{label}] box "
                              f"({b['x1']:.0f},{b['y1']:.0f})-"
                              f"({b['x2']:.0f},{b['y2']:.0f})")
                    for p in r["points"]:
                        print(f"  [{label}] point "
                              f"({p['x']:.0f},{p['y']:.0f})")
        return text

    def chat(self):
        print("""\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
=================================================================""")
        while True:
            input_str = input("\nQuestion: ")
            if input_str in ["exit", "q", "quit"]:
                break
            if input_str in ["clear", "new", "c"]:
                print("New chat session created.")
                self.model.clear_history()
                self.history_max_posid = 0
                continue
            media_path = input("\nImage Path: ")
            self.run_once(input_str, media_path)


def main(args):
    model = LocateAnything(args)
    model._config_path = args.config_path
    if args.prompt is not None:
        model.run_once(args.prompt, args.media_path)
    else:
        model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="../config",
                        help='path to the HF model config directory')
    parser.add_argument('-d', '--devid', type=int, default=0,
                        help='device ID to use')
    parser.add_argument('-p', '--prompt', type=str, default=None,
                        help='If set, run a single inference and exit.')
    parser.add_argument('--media_path', type=str, default="",
                        help='Path to an image for programmatic mode.')
    args = parser.parse_args()
    main(args)
