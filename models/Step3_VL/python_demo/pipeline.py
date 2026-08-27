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
import numpy as np


class Step3_VL():

    def __init__(self, args):
        self.device = args.devid
        self.no_think = args.no_think
        self.model = chat.Step3VL()
        self.model.init(self.device, args.model_path)
        self.processor = AutoProcessor.from_pretrained(args.config_path.rstrip("/"),
                                                       trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.ID_IM_START = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        # The image-repl block uses plain <im_start>/<im_end>/<im_patch> (no
        # pipes) to wrap the vision slots — distinct from the chat-template's
        # <|im_start|>/<|im_end|>. ViT injection is keyed off these plain tokens.
        self.ID_IMG_BLOCK_START = self.tokenizer.convert_tokens_to_ids("<im_start>")
        self.ID_IMG_BLOCK_END = self.tokenizer.convert_tokens_to_ids("<im_end>")
        self.ID_IM_PATCH = self.tokenizer.convert_tokens_to_ids("<im_patch>")
        self.ID_PATCH_START = self.tokenizer.convert_tokens_to_ids("<patch_start>")
        self.ID_PATCH_END = self.tokenizer.convert_tokens_to_ids("<patch_end>")
        self.ID_PATCH_NEWLINE = self.tokenizer.convert_tokens_to_ids("<patch_newline>")
        self.support_history = self.model.support_history

    def __del__(self):
        self.model.deinit()

    def get_media_type(self, file_path):
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext in image_exts:
            return "image"
        raise RuntimeError(f"Unsupported media type: {ext}")

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

    def process(self, messages, media_type):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=not self.no_think)
        if media_type == "text":
            return self.processor(text=[text], return_tensors="pt")
        # For image: processor handles multi-crop internally
        from PIL import Image
        image_path = None
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        image_path = item.get("image")
                        break
        image = Image.open(image_path)
        return self.processor(text=[text], images=[image], return_tensors="pt")

    def vit_process(self, inputs):
        """Process vision tokens: run ViT for the global view (and patches if
        any), then splice the ViT features into the embedding buffer at the
        positions occupied by the <im_patch> placeholder tokens.

        Token layout for one image (no patches):
            <im_start> <im_patch>x169 <im_end>   (the image-repl block)
        ViT global produces exactly 169 features, which must overwrite the 169
        <im_patch> slots — i.e. offset = block_start + 1.
        """
        input_ids = inputs.input_ids[0].tolist()

        # ---- global view ----
        pixel_values = inputs.pixel_values  # [1, 3, 728, 728]
        block_starts = [i for i, t in enumerate(input_ids)
                        if t == self.ID_IMG_BLOCK_START]
        if len(block_starts) == 0:
            print("Warning: no <im_start> image block found; skipping ViT")
            return

        # There may be multiple image blocks (one per image); Step3-VL demo
        # handles a single image, so take the first.
        global_offset = block_starts[0] + 1
        # Sanity: the next GLOBAL_TOKENS tokens must be <im_patch>.
        patch_slots = input_ids[global_offset:global_offset + self.model.GLOBAL_TOKENS]
        if patch_slots.count(self.ID_IM_PATCH) != self.model.GLOBAL_TOKENS:
            print("Warning: image block does not contain {} <im_patch> slots "
                  "(got {}); ViT injection may be misaligned".format(
                      self.model.GLOBAL_TOKENS,
                      patch_slots.count(self.ID_IM_PATCH)))
        self.model.forward_vit_global(
            pixel_values.numpy().astype(np.float32), global_offset)

        # ---- patches (multi-crop), if the processor produced any ----
        patch_pixel_values = getattr(inputs, 'patch_pixel_values', None)
        if patch_pixel_values is not None:
            num_patches = patch_pixel_values.shape[0]
            if not self.model.has_vit_patch:
                print("Warning: image produced {} patches but vit_patch "
                      "is not compiled (recompile with --max_pixels to "
                      "enable patches). Skipping.".format(num_patches))
                return
            # Each patch has its own <patch_start> in input_ids.
            # Process each patch individually at its correct offset.
            patch_start_positions = [i for i, t in enumerate(input_ids)
                                     if t == self.ID_PATCH_START]
            if len(patch_start_positions) != num_patches:
                print("Warning: {} <patch_start> tokens but {} patches; "
                      "skipping.".format(len(patch_start_positions),
                                         num_patches))
                return
            for i in range(num_patches):
                patch_offset = patch_start_positions[i] + 1
                # Sanity: next PATCH_TOKENS should be <im_patch>
                slots = input_ids[patch_offset:patch_offset + self.model.PATCH_TOKENS]
                if slots.count(self.ID_IM_PATCH) != self.model.PATCH_TOKENS:
                    print("Warning: patch {} at offset {} has {} <im_patch> "
                          "slots (expected {})".format(
                              i, patch_offset,
                              slots.count(self.ID_IM_PATCH),
                              self.model.PATCH_TOKENS))
                self.model.forward_vit_patch(
                    patch_pixel_values[i:i+1].numpy().astype(np.float32),
                    i, patch_offset)

    def forward_prefill(self, position_ids):
        if self.model.history_length == 0 or not self.support_history:
            return self.model.forward_first(position_ids)
        return self.model.forward_first(position_ids)

    def run_once(self, input_str, media_path=""):
        self.input_str = input_str
        media_path = (media_path or "").strip()
        if media_path == "":
            messages = self.text_message()
            media_type = "text"
        elif not os.path.exists(media_path):
            print("Can't find image: {}".format(media_path))
            return None
        else:
            media_type = self.get_media_type(media_path)
            messages = self.image_message(media_path)

        inputs = self.process(messages, media_type)
        token_len = inputs.input_ids.numel()
        if token_len > self.model.MAX_INPUT_LENGTH:
            print("Error: token count {} exceeds max input length {}".format(
                token_len, self.model.MAX_INPUT_LENGTH))
            return None

        print("\nAnswer:")
        first_start = time.time()

        # Embed text tokens
        self.model.forward_embed(inputs.input_ids.numpy())

        # Process vision (if image)
        vit_start = vit_end = 0
        if media_type == "image":
            vit_start = time.time()
            self.vit_process(inputs)
            vit_end = time.time()

        # Standard 1D position_ids: [0, 1, 2, ..., token_len-1]
        position_ids = np.arange(token_len, dtype=np.int32)
        max_posid = token_len - 1

        # First forward (prefill)
        token = self.forward_prefill(position_ids)
        first_end = time.time()

        tok_num = 0

        # Decode loop
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
            max_posid += 1
            position_ids = np.array([max_posid], dtype=np.int32)
            token = self.model.forward_next(position_ids)
            tok_num += 1

        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration if next_duration > 0 else 0.0
        print(f"\nFTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} tokens/s")
        if media_type == "image":
            print(f"Vision: {vit_end - vit_start:.3f} s")
        return text

    def chat(self):
        print("""\n=================================================================
1. If you want to quit, please enter one of [/q, /quit, /exit]
2. To create a new chat session, please enter one of [/clear, /new]
3. To ask about an image, include @<path> in your question
=================================================================""")
        while True:
            input_str = input("\nQuestion: ")
            if input_str in ["/exit", "/q", "/quit"]:
                break
            if input_str in ["/clear", "/new", "/c"]:
                print("New chat session created.")
                self.model.clear_history()
                continue
            input_str, media_path = extract_media(input_str)
            self.run_once(input_str, media_path)


def extract_media(input_str):
    tokens = input_str.split()
    media_paths = [t[1:] for t in tokens if t.startswith("@") and len(t) > 1]
    input_str = " ".join(t for t in tokens if not (t.startswith("@") and len(t) > 1))
    if len(media_paths) > 1:
        print("Only one media file is supported, using: {}".format(media_paths[0]))
    media_path = media_paths[0] if media_paths else ""
    return input_str, media_path


def main(args):
    model = Step3_VL(args)
    if args.prompt is not None:
        prompt, media_path = extract_media(args.prompt)
        model.run_once(prompt, media_path)
    else:
        model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="../config",
                        help='path to the processor file')
    parser.add_argument('-d', '--devid', type=int, default=0,
                        help='device ID to use')
    parser.add_argument('-p', '--prompt', type=str, default=None,
                        help='If set, run a single inference and exit. Use @<path> for image.')
    parser.add_argument('--no_think', action='store_true',
                        help='Disable thinking mode to save tokens and reduce latency.')
    args = parser.parse_args()
    main(args)
