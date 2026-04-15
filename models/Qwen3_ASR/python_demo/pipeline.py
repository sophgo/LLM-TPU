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
from transformers import AutoProcessor
import qwen_asr
import librosa


class Qwen3_ASR():

    def __init__(self, args):
        # devid
        self.device = args.devid

        # load model
        self.sample_rate = 16000
        self.language = args.language
        self.model = chat.Qwen3_ASR()
        self.model.init(self.device, args.model_path)
        self.processor = AutoProcessor.from_pretrained(args.config_path, fix_mistral_regex=True)
        self.tokenizer = self.processor.tokenizer
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.ID_AUDIO = self.tokenizer.convert_tokens_to_ids(self.processor.audio_token)
        self.ID_AUDIO_BOS = self.tokenizer.convert_tokens_to_ids(self.processor.audio_bos_token)
        self.ID_AUDIO_EOS = self.tokenizer.convert_tokens_to_ids(self.processor.audio_eos_token)

        self.support_history = self.model.support_history
        self.max_posid = 0
        self.history_max_posid = 0

    def build_text_prompt(self):
        # yapf: disable
        messages = [
            {"role": "system", "content": self.input_str},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]}
        ]
        # yapf: enable
        text = self.processor.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True)
        if self.language is not None:
            text = text + f"language {self.language}{'<asr_text>'}"
        return text

    def process(self, media_path):
        text = self.build_text_prompt()
        y, _ = librosa.load(media_path, sr=self.sample_rate)
        y = y[:(len(y) // self.sample_rate) * self.sample_rate]
        audios = [y]
        inputs = self.processor(text=text,
                                audio=audios,
                                return_tensors="pt",
                                padding=True)
        print("Inputs processed successfully.")
        return inputs

    def get_attn_mask(self, seq_length, cu_seqlens):
        attention_mask = torch.full([1, seq_length, seq_length], -10000.0, dtype=torch.float32)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
                           cu_seqlens[i - 1]:cu_seqlens[i]] = 0
        return attention_mask

    def audio_inference(self, inputs):
        start_offset = int(torch.where(inputs.input_ids == self.ID_AUDIO_BOS)[1][0])
        end_offset = int(torch.where(inputs.input_ids == self.ID_AUDIO_EOS)[1][0])
        audio_tokens_per_chunk = 13  # 25 per second * 2 seconds
        t = inputs.input_features.shape[-1]
        n_times = int(t // self.model.AUDIO_LENGTH)
        # ========= audio inference ===============================

        audio_dims = inputs.feature_attention_mask.sum().item()
        audio_features = inputs.input_features[:, :, :audio_dims].reshape(128, -1, self.model.AUDIO_LENGTH).transpose(0, 1)
        total_audio_tokens = audio_features.shape[0] * audio_tokens_per_chunk
        audio_offset = start_offset
        if audio_offset != int(torch.where(inputs.input_ids == self.ID_AUDIO_BOS)[1][0]):
            raise RuntimeError(
                f"Audio BOS token offset {audio_offset} does not match expected offset")
        if total_audio_tokens // audio_tokens_per_chunk != n_times:
            raise RuntimeError(
                f"Total audio tokens {total_audio_tokens} is not divisible by audio tokens per chunk {audio_tokens_per_chunk}"
            )
        audio_offset_list = [i * audio_tokens_per_chunk + audio_offset + 1 for i in range(n_times)]
        if (audio_features.shape[0] != n_times):
            raise RuntimeError(
                f"Audio features shape {audio_features.shape[0]} does not match expected {n_times}")
        # do audio inference
        self.model.forward_audio(audio_features.numpy(), np.array(audio_offset_list,
                                                                  dtype=np.int32))

    def get_rope_index(
        self,
        attention_mask
    ) -> torch.Tensor:
        mrope_position_deltas = []
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
        return position_ids, mrope_position_deltas

    def forward_prefill(self, position_ids):
        if self.model.history_length == 0 or not self.support_history:
            self.history_max_posid = 0
            return self.model.forward_first(position_ids)
        self.max_posid += self.history_max_posid
        position_ids = position_ids + self.history_max_posid
        return self.model.forward_first(position_ids)

    def asr(self):
        """
        Start a asr session.
        """
        # Instruct
        print("""\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new asr session, please enter one of [clear, new]
=================================================================""")
        # Stop Chatting with "exit" input
        while True:
            self.input_str = input("\nContext: ")
            # Quit
            if self.input_str in ["exit", "q", "quit"]:
                break
            if self.input_str in ["clear", "new", "c"]:
                print("New asr session created.")
                self.model.clear_history()
                self.history_max_posid = 0
                continue

            media_path = input("\nAudio Path: ")
            media_path = media_path.strip()
            if media_path == "":
                print("Error: No input, try again!!")
                continue
            elif not os.path.exists(media_path):
                print("Can't find audio: {}".format(media_path))
                continue
            inputs = self.process(media_path)
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
            position_ids, rope_deltas = self.get_rope_index(inputs.attention_mask)
            self.max_posid = int(position_ids.max())
            print("\nResult:")

            # Chat
            first_start = time.time()
            self.model.forward_embed(inputs.input_ids.squeeze(0).tolist())
            self.audio_inference(inputs)

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
                    if word == "<asr_text>":
                        text = ""
                        word = "\n"
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
    model = Qwen3_ASR(args)
    model.asr()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="../config",
                        help='path to the processor file')
    parser.add_argument('-d', '--devid', type=int, default=0, help='device ID to use')
    parser.add_argument('-l', '--language', type=str, default=None, help='forced language')
    # yapf: enable
    args = parser.parse_args()
    main(args)
