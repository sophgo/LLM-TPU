# ==============================================================================
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
#
# Mage-VL pipeline: phase-1 offline VLM + phase-4 streaming gate.
#
# Text backbone is plain Qwen3-4B (1D RoPE), so the LLM position_ids are a
# simple range(0, token_len). The 3D RoPE lives INSIDE the vit net, fed by the
# per-patch (t, h, w) grid coordinates built in rot_pos_3d(). Image embeddings
# (merger output, [N/4, HIDDEN_SIZE]) are injected into dev_buffer at the
# image_pad span (right after <|vision_start|>), overwriting the placeholder
# token embeddings produced by forward_embed.
#
# Streaming mode (phase 4): for video input, the pipeline loads ALL frames,
# divides them into segments of GATE_FRAMES, runs ViT + Gate + ClsNet per
# segment to decide "silent" vs "speak". On a "speak" decision the LLM is
# invoked independently to generate a description of that segment.

import time
import argparse
import os
import numpy as np
from PIL import Image
from transformers import AutoProcessor
import chat
from magevl_video import load_video_frames


def _force_trust_remote_code():
    """Force transformers' ``resolve_trust_remote_code`` to True everywhere.

    transformers 5.7 does not forward ``trust_remote_code=True`` from
    ``AutoProcessor.from_pretrained`` down to the dynamic-module loader that
    pulls in the custom ``MageVLProcessor`` class, so it raises an interactive
    [y/N] prompt. In a non-interactive TPU demo that prompt blocks on stdin
    (and the SIGALRM fallback is unsafe with native extensions loaded), so we
    resolve it to True up front. The custom code is our own config/ files.
    """
    import importlib
    import transformers.dynamic_module_utils as dmu

    def _always_true(*args, **kwargs):
        return True

    dmu.resolve_trust_remote_code = _always_true
    for name in (
        "transformers.models.auto.processing_auto",
        "transformers.models.auto.image_processing_auto",
        "transformers.models.auto.tokenization_auto",
        "transformers.models.auto.configuration_auto",
        "transformers.models.auto.video_processing_auto",
        "transformers.models.auto.feature_extraction_auto",
        "transformers.models.auto.auto_factory",
    ):
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "resolve_trust_remote_code"):
                mod.resolve_trust_remote_code = _always_true
        except Exception:
            pass


_force_trust_remote_code()



class Mage_VL():

    def __init__(self, args):
        self.device = args.devid

        # load model
        self.model = chat.Mage_VL()
        self.model.init(self.device, args.model_path)
        self.processor = AutoProcessor.from_pretrained(
            args.config_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        # special token ids — read from the tokenizer, never hard-code.
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.ID_END = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.ID_VISION_START = \
            self.tokenizer.convert_tokens_to_ids("<|vision_start|>")

        # The vit net is STATIC (num_patches == MAX_PATCHES). The Qwen2VL
        # smart_resize does NOT pin the patch count (flooring by aspect ratio
        # gives e.g. 352 instead of 392), so we disable the processor's resize
        # and pre-resize each image ourselves to a grid of exactly MAX_PIXELS
        # pixels (H*W = MAX_PIXELS, both multiples of 32 => num_patches =
        # MAX_PATCHES regardless of which (H,W) split is chosen).
        self.processor.image_processor.do_resize = False
        self.merge_size = self.processor.image_processor.merge_size  # 2

        # Number of frames sampled per video. The Mage-VL frames backend
        # treats each frame as an independent single-frame image (no
        # cross-frame attention inside the vit), so T is a free knob: more
        # frames = more temporal coverage at the cost of T vit calls and
        # T*MAX_PATCHES vision tokens injected into the LM context.
        self.num_video_frames = getattr(args, "num_video_frames", 4) or 4

    def __del__(self):
        self.model.deinit()

    @property
    def has_gate(self):
        """True when the loaded bmodel contains gate + cls_net streaming nets."""
        return self.model.GATE_FRAMES > 0

    def text_message(self):
        # yapf: disable
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": self.input_str}],
        }]
        # yapf: enable
        return messages

    def image_message(self, path):
        # Mage-VL chat template emits <|vision_start|><|image_pad|><|vision_end|>
        # for a {"type": "image"} content item; the actual PIL image is passed
        # to the processor separately (see process()).
        # yapf: disable
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": self.input_str},
            ],
        }]
        # yapf: enable
        return messages

    def video_message(self, path):
        # Mage-VL chat template emits <|vision_start|><|video_pad|><|vision_end|>
        # for a {"type": "video"} content item; the processor's frames backend
        # rewrites <|video_pad|> into per-frame `<X.X seconds>`+<|image_pad|>
        # blocks. The decoded PIL frames are passed to the processor separately
        # (see process()).
        # yapf: disable
        messages = [{
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": self.input_str},
            ],
        }]
        # yapf: enable
        return messages

    def get_media_type(self, file_path):
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.flv'}
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext in image_exts:
            return "image"
        if ext in video_exts:
            return "video"
        raise RuntimeError(f"Unsupported media type: {ext}")

    def resize_to_fixed_pixels(self, img):
        """Resize a PIL image to a grid of exactly MAX_PIXELS pixels.

        Picks (H, W) both multiples of (patch_size * merge_size) with
        H*W == MAX_PIXELS that best matches the source aspect ratio, so the
        vit net always receives num_patches == MAX_PATCHES tokens.
        """
        import math
        factor = 16 * self.merge_size  # 32
        n_units = self.model.MAX_PIXELS // (factor * factor)  # 98
        src_w, src_h = img.size
        ratio = src_h / src_w
        best = None
        best_diff = 1e9
        for a in range(1, int(math.isqrt(n_units)) + 1):
            if n_units % a:
                continue
            b = n_units // a
            for hh, ww in ((a, b), (b, a)):
                diff = abs((hh / ww) - ratio)
                if diff < best_diff:
                    best_diff = diff
                    best = (hh * factor, ww * factor)  # (H, W)
        H, W = best
        return img.resize((W, H), Image.BILINEAR)

    def rot_pos_3d(self, grid_thw):
        """Per-patch (t, h, w) grid coordinates for the vit's 3D RoPE.

        Kept for reference; the live path uses the processor's
        ``patch_positions`` (see vit_process_image), which is built by
        ``build_patch_positions`` and matches this layout for a single image
        while also carrying the correct per-frame t-axis for video.
        """
        t, h, w = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
        ms = self.merge_size  # 2
        mh, mw = h // ms, w // ms
        # one-frame block-ordered (h, w); C-order flatten => bh, bw, dh, dw
        bh, bw, dh, dw = np.mgrid[0:mh, 0:mw, 0:ms, 0:ms]
        h_pos = (bh * ms + dh).reshape(-1).astype(np.int32)
        w_pos = (bw * ms + dw).reshape(-1).astype(np.int32)
        # tile across t frames
        pos_h = np.tile(h_pos, t)
        pos_w = np.tile(w_pos, t)
        pos_t = np.repeat(np.arange(t, dtype=np.int32), h_pos.size)
        return pos_t, pos_h, pos_w

    def vit_process_image(self, inputs):
        # <|vision_start|> sits immediately before the expanded image_pad span;
        # injection starts one position after it.
        vit_token_list = torch_where(inputs.input_ids == self.ID_VISION_START)
        # The processor emits a single contiguous patch_positions tensor
        # [N_total, 3] in (t, h, w) block layout covering ALL visuals (one
        # row per patch, frame-by-frame for video). We slice it per visual
        # the same way we slice pixel_values: by each row of image_grid_thw.
        patch_positions = inputs.patch_positions.numpy().astype(np.int32)
        pre_patches = 0
        for idx, vit_offset in enumerate(vit_token_list):
            grid_thw = inputs.image_grid_thw[idx]
            num_patches = int(grid_thw[0]) * int(grid_thw[1]) * int(grid_thw[2])
            hidden_states = inputs.pixel_values[pre_patches:
                                                pre_patches + num_patches, :]
            pos = patch_positions[pre_patches:pre_patches + num_patches]
            pos_t, pos_h, pos_w = pos[:, 0], pos[:, 1], pos[:, 2]
            self.model.forward_vit(hidden_states.numpy(), pos_t, pos_h,
                                    pos_w, vit_offset + 1)
            pre_patches += num_patches

    def process(self, messages, media_type):
        if media_type == "text":
            return self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt")
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        if media_type == "image":
            img = Image.open(self.media_path).convert("RGB")
            img = self.resize_to_fixed_pixels(img)
            return self.processor(text=[text],
                                  images=[img],
                                  return_tensors="pt")
        if media_type == "video":
            # Decode T frames off the video file and resize each to the fixed
            # pixel budget (do_resize is off, so the processor skips its own
            # smart_resize and each frame yields exactly MAX_PATCHES patches).
            # The frames-backend video path treats each frame as an
            # independent single-frame image: image_grid_thw is expanded to
            # T rows [1, H, W] and patch_positions carries per-frame t-indices
            # (0..T-1), so the vit encodes each frame in its own bidirectional
            # attention chunk - exactly what the static 392-patch vit bmodel
            # expects when called once per frame.
            frames = load_video_frames(self.media_path,
                                        self.num_video_frames,
                                        self.resize_to_fixed_pixels)
            return self.processor(text=[text],
                                  videos=[frames],
                                  video_backend="frames",
                                  return_tensors="pt")
        raise RuntimeError(f"Unsupported media type: {media_type}")

    def run_once(self, input_str, media_path=""):
        """Run a single inference turn; returns the generated text."""
        self.input_str = input_str
        media_path = (media_path or "").strip()
        if media_path == "":
            messages = self.text_message()
            media_type = "text"
        elif not os.path.exists(media_path):
            print("Can't find media: {}".format(media_path))
            return None
        else:
            media_type = self.get_media_type(media_path)
            self.media_path = media_path
            if media_type == "image":
                messages = self.image_message(media_path)
            elif media_type == "video":
                messages = self.video_message(media_path)
            else:
                print("Unsupported media type: {}".format(media_path))
                return None

        inputs = self.process(messages, media_type)
        token_len = inputs.input_ids.numel()
        if token_len > self.model.MAX_INPUT_LENGTH:
            print("Error: input length {} exceeds MAX_INPUT_LENGTH {}.".format(
                token_len, self.model.MAX_INPUT_LENGTH))
            return None
        print("\nAnswer:")

        # 1) text embeddings -> dev_buffer
        first_start = time.time()
        self.model.forward_embed(inputs.input_ids.numpy().reshape(-1))
        # 2) vision: inject image/video embeddings into dev_buffer. Video
        # injects T frame embeddings (one forward_vit call per frame) at the
        # T successive <|vision_start|> positions the processor emitted.
        vit_start = vit_end = 0.0
        if media_type in ("image", "video"):
            vit_start = time.time()
            self.vit_process_image(inputs)
            vit_end = time.time()
        # 3) prefill (1D position_ids = range(token_len))
        position_ids = np.arange(token_len, dtype=np.int32)
        token = self.model.forward_first(position_ids)
        first_end = time.time()

        # 4) decode loop (forward_next computes 1D position_id internally)
        tok_num = 0
        full_word_tokens = []
        text = ""
        gen_tokens = [int(token)]
        while token not in [self.ID_IM_END, self.ID_END
                           ] and self.model.history_length < self.model.SEQLEN:
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens,
                                         skip_special_tokens=True)
            if "�" not in word:
                if len(full_word_tokens) == 1:
                    pre_word = word
                    word = self.tokenizer.decode([token, token],
                                                 skip_special_tokens=True
                                                 )[len(pre_word):]
                text += word
                print(word, flush=True, end="")
                full_word_tokens = []
            token = self.model.forward_next()
            gen_tokens.append(int(token))
            tok_num += 1
        next_end = time.time()

        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration if next_duration > 0 else 0.0
        print(f"\nFTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} tokens/s")
        if media_type in ("image", "video"):
            print(f"Vision: {vit_end - vit_start:.3f} s")
        return text

    @staticmethod
    def should_speak(logits, threshold=0.0):
        """Decide whether the gate says "speak" for a segment.

        ``logits`` is the flat list returned by ``forward_gate``: for each
        frame *t*, ``logits[2*t]`` is the silent score and ``logits[2*t+1]``
        the speak score.  Decision: average across frames, then argmax.

        Returns:
            (bool, float) — (should_speak, speak_margin).  Margin is the
            mean speak score minus the mean silent score (positive = speak).
        """
        n = len(logits) // 2
        silent = np.mean([logits[2 * i] for i in range(n)])
        speak = np.mean([logits[2 * i + 1] for i in range(n)])
        return (speak - silent) > threshold, speak - silent

    def run_streaming(self, input_str, media_path, threshold=0.0):
        """Process a video in streaming fashion with gate decisions.

        Reads frames lazily from the video (via PyAV), buffering only
        ``GATE_FRAMES`` frames at a time. Each segment runs ViT + Gate +
        ClsNet, and triggers LLM generation on "speak" decisions.

        Args:
            input_str: the text prompt used for every generation event.
            media_path: path to the video file.
            threshold: speak margin (default 0 = argmax).
        """
        import av

        if not self.has_gate:
            print("Error: loaded bmodel has no gate/cls_net (GATE_FRAMES=0).")
            return

        T = self.model.GATE_FRAMES
        if not os.path.exists(media_path):
            print(f"Can't find media: {media_path}")
            return

        print(f"Opening video: {media_path}")
        container = av.open(media_path)

        speak_count = 0
        seg_idx = 0
        frame_buf = []
        frame_count = 0

        try:
            for av_frame in container.decode(video=0):
                img = av_frame.to_image().convert("RGB")
                img = self.resize_to_fixed_pixels(img)
                frame_buf.append(img)
                frame_count += 1

                if len(frame_buf) < T:
                    continue

                # --- We have T frames, process one segment ---
                seg_idx += 1
                frames = frame_buf
                frame_buf = []  # release reference so GC can reclaim

                # Build video message and process with the processor to get
                # the correct input_ids / patch_positions for T frames.
                self.input_str = input_str
                self.media_path = media_path
                messages = self.video_message(media_path)
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text],
                                        videos=[frames],
                                        video_backend="frames",
                                        return_tensors="pt")

                # --- ViT + read embeddings ---
                vit_t0 = time.time()
                self.model.forward_embed(
                    inputs.input_ids.numpy().reshape(-1))
                self.vit_process_image(inputs)
                vit_t1 = time.time()

                # Read per-frame merged embeddings [98, 2560] f32, mean-pool.
                vit_token_list = torch_where(
                    inputs.input_ids == self.ID_VISION_START)
                frame_embs = []
                for idx, vit_offset in enumerate(vit_token_list):
                    grid_thw = inputs.image_grid_thw[idx]
                    num_patches = int(grid_thw[0]) * int(grid_thw[1]) * int(
                        grid_thw[2])
                    num_merged = num_patches // (self.merge_size * self.merge_size)
                    embs = self.model.read_vit_embeddings(vit_offset + 1,
                                                          num_merged)
                    frame_embs.append(np.array(embs).mean(axis=0))

                averaged = np.array(frame_embs, dtype=np.float32)  # [T, 2560]

                # --- Gate decision ---
                gate_t0 = time.time()
                logits = self.model.forward_gate(averaged)
                speak, margin = self.should_speak(logits, threshold)
                gate_t1 = time.time()

                status = "SPEAK" if speak else "SILENT"
                print(f"[Seg {seg_idx:>3}] frame #{frame_count}  "
                      f"ViT {vit_t1 - vit_t0:.3f}s  "
                      f"Gate {gate_t1 - gate_t0:.3f}s  "
                      f"→ {status} (margin {margin:+.4f})")

                if not speak:
                    continue

                # --- LLM generation (forward_embed → forward_first → decode) ---
                speak_count += 1
                self.model.clear_history()
                self.model.forward_embed(
                    inputs.input_ids.numpy().reshape(-1))
                self.vit_process_image(inputs)

                token_len = inputs.input_ids.numel()
                position_ids = np.arange(token_len, dtype=np.int32)
                token = self.model.forward_first(position_ids)

                tok_num = 0
                full_word_tokens = []
                text_out = ""
                gen_t0 = time.time()
                while token not in [self.ID_IM_END, self.ID_END
                                    ] and self.model.history_length < self.model.SEQLEN:
                    full_word_tokens.append(token)
                    word = self.tokenizer.decode(full_word_tokens,
                                                 skip_special_tokens=True)
                    if "�" not in word:
                        if len(full_word_tokens) == 1:
                            pre_word = word
                            word = self.tokenizer.decode(
                                [token, token],
                                skip_special_tokens=True)[len(pre_word):]
                        text_out += word
                        print(word, flush=True, end="")
                        full_word_tokens = []
                    token = self.model.forward_next()
                    tok_num += 1
                gen_t1 = time.time()

                tps = tok_num / (gen_t1 - gen_t0) if (gen_t1 - gen_t0) > 0 else 0
                print(f"\n  [{speak_count}] {tok_num} tokens, "
                      f"{tps:.1f} tok/s")

        finally:
            container.close()

        print(f"\nDone. {frame_count} frames, {seg_idx} segments, "
              f"{speak_count} spoke.")

    def chat(self):
        streaming = False
        gate_threshold = 0.0
        print("""\n=================================================================
1. If you want to quit, please enter one of [/q, /quit, /exit]
2. To create a new chat session, please enter one of [/clear, /new]
3. To ask about an image, include @<path> in your question
4. /stream          — toggle streaming mode for video input
5. /threshold <val> — set gate speak-margin threshold (default 0.0)
=================================================================""")
        while True:
            input_str = input("\nQuestion: ")
            if input_str in ["/exit", "/q", "/quit"]:
                break
            if input_str in ["/clear", "/new", "/c"]:
                print("New chat session created.")
                self.model.clear_history()
                continue
            if input_str.strip() == "/stream":
                if not self.has_gate:
                    print("Streaming not available (bmodel has no gate).")
                    continue
                streaming = not streaming
                print(f"Streaming mode: {'ON' if streaming else 'OFF'}")
                continue
            if input_str.strip().startswith("/threshold"):
                parts = input_str.strip().split()
                if len(parts) == 2:
                    try:
                        gate_threshold = float(parts[1])
                        print(f"Gate threshold: {gate_threshold:+.4f}")
                    except ValueError:
                        print("Usage: /threshold <float>")
                else:
                    print("Usage: /threshold <float>")
                continue
            input_str, media_path = extract_media(input_str)
            if streaming and media_path and os.path.exists(media_path):
                try:
                    media_type = self.get_media_type(media_path)
                except RuntimeError:
                    media_type = None
                if media_type == "video":
                    self.run_streaming(input_str, media_path, gate_threshold)
                    continue
            self.run_once(input_str, media_path)


def torch_where(mask):
    """Return the indices of True elements in a 1-D/2-D torch bool tensor as a
    python list. Kept tiny so the pipeline does not need torch except via the
    processor output."""
    import torch
    return torch.where(mask)[1].tolist() if mask.dim() > 1 else \
        torch.where(mask)[0].tolist()


def extract_media(input_str):
    """Split @<path> media attachments out of the input text."""
    tokens = input_str.split()
    media_paths = [t[1:] for t in tokens if t.startswith("@") and len(t) > 1]
    input_str = " ".join(
        t for t in tokens if not (t.startswith("@") and len(t) > 1))
    if len(media_paths) > 1:
        print("Only one media file is supported, using: {}".format(
            media_paths[0]))
    media_path = media_paths[0] if media_paths else ""
    return input_str, media_path


def main(args):
    model = Mage_VL(args)
    if args.prompt is not None:
        prompt, media_path = extract_media(args.prompt)
        if args.stream:
            if not model.has_gate:
                print("Error: --stream requires a bmodel with gate/cls_net.")
                return
            model.run_streaming(prompt, media_path, args.gate_threshold)
        else:
            model.run_once(prompt, media_path)
    else:
        model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="../config",
                        help='path to the processor config directory')
    parser.add_argument('-d', '--devid', type=int, default=0,
                        help='device ID to use')
    parser.add_argument('-p', '--prompt', type=str, default=None,
                        help='If set, run programmatically (non-interactive): '
                             'a single inference is performed using this prompt '
                             'and then the program exits. Include @<path> to '
                             'attach an image or video.')
    parser.add_argument('--num_video_frames', type=int, default=4,
                        help='Number of frames sampled per video (default 4). '
                             'Each frame is one vit call (T*MAX_PATCHES vision '
                             'tokens injected into the LM context).')
    parser.add_argument('--stream', action='store_true',
                        help='Enable streaming mode for video input. The '
                             'video is divided into segments of GATE_FRAMES '
                             'frames; each segment gets a gate decision, and '
                             'the LLM generates only when "speak" is chosen. '
                             'Requires a bmodel compiled with gate+cls_net.')
    parser.add_argument('--gate_threshold', type=float, default=0.0,
                        help='Speak-margin threshold for streaming gate '
                             'decision (default 0.0 = argmax). Higher values '
                             'require more confidence to speak.')
    # yapf: enable
    args = parser.parse_args()
    main(args)
