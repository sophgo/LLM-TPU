import argparse

import chat
import torch
import numpy as np
import time
import os
import librosa
import ffmpeg
from transformers import AutoProcessor, AutoConfig, GenerationConfig

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}


class Gemma4():

    def __init__(self, args):
        # devid
        self.devid = int(args.devid)

        # load tokenizer
        print("Load " + args.config_path + " ...")
        self.processor = AutoProcessor.from_pretrained(args.config_path, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(args.config_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        # warm up
        self.tokenizer.decode([0])

        # EOS tokens (Gemma4 has multiple eos: [1, 106])
        self.EOS = self.config.eos_token_id
        if isinstance(self.EOS, int):
            self.EOS = [self.EOS]

        self.ID_IMAGE_PAD = self.config.image_token_id
        self.ID_VIDEO_PAD = self.config.video_token_id
        self.ID_AUDIO_PAD = self.config.audio_token_id
        self.ID_BOA = self.config.boa_token_id
        self.ID_EOA = self.config.eoa_token_id
        self.mm_tokens_per_image = self.config.vision_soft_tokens_per_image

        # system prompt
        self.system_prompt = {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant."
            }]
        }

        self.model = chat.Gemma4()
        self.init_params(args)
        self.load_model(args.model_path)
        self.model.SLIDING_WINDOW = self.config.text_config.sliding_window
        self.model.ID_IMAGE_PAD = self.ID_IMAGE_PAD
        self.model.ID_VIDEO_PAD = self.ID_VIDEO_PAD
        self.model.ID_AUDIO_PAD = self.ID_AUDIO_PAD
        self.model.ID_BOA = self.ID_BOA
        self.model.ID_EOA = self.ID_EOA

    def __del__(self):
        self.model.deinit()

    def load_model(self, model_path):
        load_start = time.time()
        self.model.init(self.devid, model_path)
        load_end = time.time()
        print(f"\nLoad Time: {(load_end - load_start):.3f} s")

    def init_params(self, args):
        self.model.generation_mode = "greedy"
        if args.do_sample:
            gen_config = GenerationConfig.from_pretrained(args.config_path)
            self.model.generation_mode = "sample"
            self.model.temperature = gen_config.temperature
            self.model.top_p = gen_config.top_p
            self.model.top_k = gen_config.top_k
            self.model.penalty = gen_config.repetition_penalty
            for i in gen_config.eos_token_id:
                if i not in self.EOS:
                    self.EOS.append(i)

    def text_message(self):
        messages = [
            self.system_prompt,
            {
                "role": "user",
                "content": [{"type": "text", "text": self.input_str}],
            }]
        return messages

    def image_message(self, path):
        messages = [
            self.system_prompt,
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": path},
                    {"type": "text", "text": self.input_str}],
            }]
        return messages

    def video_message(self, path):
        messages = [
            self.system_prompt,
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": path},
                    {"type": "text", "text": self.input_str}],
            }]
        return messages

    def audio_message(self, path):
        messages = [
            # self.system_prompt,
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": path},
                    {"type": "text", "text": self.input_str}],
            }]
        return messages

    def get_media_type(self, file_path):
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext in IMAGE_EXTS:
            return "image"
        if ext in VIDEO_EXTS:
            return "video"
        if ext in AUDIO_EXTS:
            return "audio"
        raise RuntimeError(f"Unsupported media type: {ext}")

    def has_audio(self, media_path):
        try:
            info = ffmpeg.probe(media_path)
            return any(s.get("codec_type") == "audio" for s in info["streams"])
        except Exception:
            return False

    def vit_process_image(self, inputs):
        """Process image through vit and merge into embeddings."""
        input_ids = inputs.input_ids.flatten()
        vit_token_list = torch.where(input_ids == self.ID_IMAGE_PAD)[0].tolist()
        if not vit_token_list:
            return
        pixel_values = inputs.pixel_values
        position_ids = inputs.image_position_ids
        vit_offsets = [vit_token_list[0]]
        tokens_to_copy = len(vit_token_list)
        self.model.forward_vit(pixel_values.numpy(), position_ids.numpy(),
                                vit_offsets, tokens_to_copy)

    def vit_process_video(self, inputs):
        """Process video through vit_video bmodel (batch=32).
        pixel_values_videos: [1, num_frames, patches_per_frame, patch_dim]
        video_position_ids: [1, num_frames, patches_per_frame, 2]
        HF flatten(0,1) then batch inference, we do the same.
        Video tokens may not be contiguous across frames, so detect frame
        boundaries from gaps in vit_token_list.
        """
        input_ids = inputs.input_ids.flatten()
        vit_token_list = torch.where(input_ids == self.ID_VIDEO_PAD)[0].tolist()
        if not vit_token_list:
            return

        pixel_values_videos = inputs.pixel_values_videos
        video_position_ids = inputs.video_position_ids

        # Flatten [1, num_frames, ...] → [num_frames, ...] for batch inference
        frame_pixels = pixel_values_videos.flatten(0, 1)
        frame_pos_ids = video_position_ids.flatten(0, 1)
        num_frames = frame_pixels.shape[0]

        # Find frame start offsets by detecting gaps in vit_token_list.
        # Within a frame, video tokens are contiguous (diff=1).
        # Between frames, there's a gap (timestamp, boi, eoi tokens).
        vit_offsets = [vit_token_list[0]]
        for i in range(1, len(vit_token_list)):
            if vit_token_list[i] - vit_token_list[i - 1] != 1:
                vit_offsets.append(vit_token_list[i])
        tokens_per_frame = len(vit_token_list) // len(vit_offsets)

        assert len(vit_offsets) == num_frames, \
            f"Expected {num_frames} frame boundaries, found {len(vit_offsets)}"

        self.model.forward_vit(frame_pixels.numpy(), frame_pos_ids.numpy(),
                                vit_offsets, tokens_per_frame, is_video=True)

    def audio_process(self, inputs):
        """Process audio through static audio bmodel and merge into embeddings.
        Audio mel features are padded to the bmodel's compiled max_audio_mel shape,
        processed in a single pass. Only actual_audio_seq_len tokens are copied back.
        """
        input_ids = inputs.input_ids.flatten()
        audio_token_list = torch.where(input_ids == self.ID_AUDIO_PAD)[0].tolist()
        if not audio_token_list:
            return

        audio_features = inputs.input_features
        audio_mask = inputs.input_features_mask.float()

        # Trim to valid mel frames using mask
        audio_dims = int(audio_mask.sum(dim=-1).item())
        audio_features = audio_features[:, :, :audio_dims]

        # Find the start of audio tokens (first token after boa marker)
        boa_positions = torch.where(input_ids == self.ID_BOA)[0].tolist()
        audio_start = boa_positions[0] + 1  # first audio token after boa

        # Compute actual audio token count after subsample (2 stride-2 Conv2d)
        actual_audio_seq_len = audio_dims // 4

        # Check if actual audio exceeds the compiled audio_length
        max_audio_tokens = self.model.AUDIO_TOKENS
        if actual_audio_seq_len > max_audio_tokens:
            print(f"Audio length {actual_audio_seq_len} tokens exceeds compiled max {max_audio_tokens}, "
                  f"please recompile with a larger --audio_length")
            actual_audio_seq_len = max_audio_tokens
            audio_dims = max_audio_tokens * 4
            audio_features = audio_features[:, :audio_dims]

        # Pad audio mel features to bmodel's compiled input shape
        max_audio_mel = self.model.AUDIO_MEL
        padded_audio = np.zeros([1, 1, max_audio_mel, 128], dtype=np.float32)
        padded_audio[:, :, :audio_dims, :] = audio_features.numpy()

        # audio_offset: [audio_start, actual_audio_seq_len]
        audio_offset = np.array([audio_start, actual_audio_seq_len], dtype=np.int32)
        self.model.forward_audio(padded_audio, audio_offset)

    def chat(self):
        print("""\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
=================================================================""")
        while True:
            self.input_str = input("\nQuestion: ")
            # input() doesn't interpret escape sequences, so \n becomes literal \\n
            self.input_str = self.input_str.replace("\\n", "\n").replace("\\t", "\t")
            if self.input_str in ["exit", "q", "quit"]:
                break
            if self.input_str in ["clear", "new"]:
                continue

            media_path = input("\nImage, Video, or Audio Path: ")
            media_path = media_path.strip()
            if media_path == "":
                messages = self.text_message()
                media_type = "text"
                has_audio = False
            elif not os.path.exists(media_path):
                print("Can't find file: {}".format(media_path))
                continue
            else:
                has_audio = False
                try:
                    media_type = self.get_media_type(media_path)
                except RuntimeError as e:
                    print(e)
                    continue
                if media_type == "image":
                    messages = self.image_message(media_path)
                elif media_type == "video":
                    if self.has_audio(media_path):
                        has_audio = True
                    messages = self.video_message(media_path)
                elif media_type == "audio":
                    has_audio = True
                    messages = self.audio_message(media_path)

            # Load audio waveform if needed
            audio_waveform = None
            if has_audio:
                waveform, _ = librosa.load(media_path, sr=16000)
                # waveform = waveform[:128128]  # Trim or pad to 8.008s at 16kHz (max_audio_mel * 4)
                audio_waveform = waveform

            # Process inputs
            if has_audio:
                # Two-step approach: template → processor with audio waveforms
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(
                    text=text, audio=[audio_waveform],
                    return_tensors="pt", padding=True)
            else:
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True)
            token_len = inputs.input_ids.numel()
            if token_len >= self.model.MAX_INPUT_LENGTH:
                print(
                    "The maximum question length should be shorter than {} but we get {} instead."
                    .format(self.model.MAX_INPUT_LENGTH, token_len))
                continue

            print("\nAnswer:")
            first_start = time.time()
            self.model.forward_embed(inputs.input_ids.flatten().tolist())
            if media_type == "image":
                self.vit_process_image(inputs)
            elif media_type == "video":
                self.vit_process_video(inputs)
            if has_audio:
                self.audio_process(inputs)

            token = self.model.forward_first()
            first_end = time.time()
            tok_num = 0
            # Following tokens
            full_word_tokens = []
            text = ""
            while token not in self.EOS and self.model.token_length < self.model.SEQLEN:
                # breakpoint()
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

                token = self.model.forward_next()
                tok_num += 1
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")


def main(args):
    model = Gemma4(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="../config", help='path to the tokenizer file')
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--do_sample', action='store_true', help="if set, generate tokens by sample parameters")
    args = parser.parse_args()
    main(args)
