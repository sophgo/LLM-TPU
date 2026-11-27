# ==============================================================================
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import time
import argparse
from transformers import AutoProcessor, AutoConfig
import chat
import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from utils import _merge_input_ids_with_audio_features, _get_feat_extract_output_lengths 

import librosa
from io import BytesIO
from urllib.request import urlopen

from utils import to_numpy



class GreedyHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, m_logits):
        _, token = torch.topk(m_logits.float(), 1)
        return token


class Qwen2Audio():

    def __init__(self, args):
        # devid
        self.device = args.devid
        self.config_path = args.config_path

        # load model
        self.model = chat.Qwen2Audio()
        self.model.init(self.device, args.model_path)
        self.processor = AutoProcessor.from_pretrained(self.config_path, 
                                                       trust_remote_code=True)
        
        self.tokenizer = self.processor.tokenizer
        self.config = AutoConfig.from_pretrained(self.config_path, trust_remote_code=True)
        self.ID_END = self.tokenizer.convert_tokens_to_ids("<|end|>")
        self.ID_AU_END = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

        self.NUM_LAYERS = self.model.NUM_LAYERS

    def audio_message(self, text_path, audio_path):
        conversation = [
            {"role": "assistant", "content": "Yes, the speaker is female and in her twenties."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": text_path},
                ]},
        ]
        return conversation

    def process(self, conversation):
        text = self.processor.apply_chat_template(conversation,
                                                  tokenize=False,
                                                  add_generation_prompt=True)
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(librosa.load(
                            ele['audio_url'],
                            sr=self.processor.feature_extractor.sampling_rate)[0]
                        )
        #for message in conversation:
        #    if isinstance(message["content"], list):
        #        for ele in message["content"]:
        #            if ele["type"] == "audio":
        #                audios.append(librosa.load(
        #                    BytesIO(urlopen(ele['audio_url']).read()),
        #                    sr=self.processor.feature_extractor.sampling_rate)[0]
        #                )
        inputs = self.processor(
            text=text,
            audios=audios,
            return_tensors="pt",
        )

        return inputs

    def chat(self):
        """
        Start a chat session.
        """
        # Instruct
        print("""\n=================================================================
                1. If you want to quit, please enter one of [q, quit, exit]
                2. To create a new chat session, please enter one of [clear, new]
                =================================================================""")
        
        greedy = GreedyHead()
        # Stop Chatting with "exit" input
        while True:
            self.input_str = input("\nQuestion: ")
            # Quit
            if self.input_str in ["exit", "q", "quit"]:
                break
            
            media_path = input("\naudios path: ")
            if media_path == '':
                media_path = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav'
            if self.input_str == '':
                self.input_str = "这句话是什么意思"
            media_path = media_path.strip()
            messages = self.audio_message(self.input_str, media_path)
            inputs = self.process(messages)
            ###process inputs
            for k,v in inputs.items():
                inputs[k] = v  # dict_keys(['input_ids', 'attention_mask', 'input_features', 'feature_attention_mask'])

            input_ids_shape = inputs['input_ids'].shape[-1]
            input_ids = torch.ones(1, 599).to(inputs['input_ids'].dtype)
            attention_mask = torch.zeros(1, 599).to(inputs['attention_mask'].dtype)
            input_ids[..., :input_ids_shape] = inputs['input_ids']
            attention_mask[..., :input_ids_shape] = inputs['attention_mask']

            input_features = inputs['input_features']
            feature_attention_mask = inputs['feature_attention_mask']
            # Chat
            first_start = time.time()
            inputs_embeds = self.model.forward_embed(to_numpy(inputs['input_ids']).astype(np.int32))
            inputs_embeds = inputs_embeds.reshape((1,599,4096))
            inputs_embeds = torch.from_numpy(inputs_embeds)
            
            ##### get feature lengths 
            audio_feat_lengths, audio_output_lengths = _get_feat_extract_output_lengths(
                    feature_attention_mask.sum(-1)
                )
            batch_size, _, max_mel_seq_len = input_features.shape
            max_seq_len = (max_mel_seq_len - 2) // 2 + 1
            # Create a sequence tensor of shape (batch_size, max_seq_len)
            seq_range = (
                            torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
                            .unsqueeze(0)
                            .expand(batch_size, max_seq_len)
                        )
            lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
            # Create mask
            padding_mask = seq_range >= lengths_expand

            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                            batch_size, 1, max_seq_len, max_seq_len
                        )
            audio_attention_mask = audio_attention_mask_.to(
                            dtype=torch.float32, device="cpu"
                        )
            audio_attention_mask[audio_attention_mask_] = float("-inf")

            ## forward audio
            audio_outputs = self.model.forward_audio(to_numpy(inputs['input_features']), to_numpy(audio_attention_mask))

            ## forward project
            audio_outputs = audio_outputs.reshape((1, 750, 1280))
            audio_features = self.model.forward_project(audio_outputs)

            audio_features = audio_features.reshape((1, 750, 4096))
            audio_features = torch.from_numpy(audio_features)

            inputs_embeds, attention_mask, label, position_ids, _ = _merge_input_ids_with_audio_features(
                audio_features, audio_output_lengths, inputs_embeds, input_ids[..., :input_ids_shape], attention_mask[..., :input_ids_shape], None, self.config
            )

            ## prefill 
            attention_mask_prefill = torch.zeros(1, 599).to(inputs['attention_mask'].dtype)
            attention_mask_prefill[..., :attention_mask.shape[-1]] = attention_mask
            inputs_embeds_prefill = torch.zeros(1, 599, 4096).to(inputs_embeds.dtype)
            inputs_embeds_prefill[:, :inputs_embeds.shape[-2], :] = inputs_embeds
            position_ids_prefill = torch.zeros(1, 599).to(position_ids.dtype)
            position_ids_prefill[..., :position_ids.shape[-1]] = position_ids
            attention_mask = attention_mask_prefill
            inputs_embeds  = inputs_embeds_prefill
            position_ids   = position_ids_prefill

            inputs_embeds = to_numpy(inputs_embeds)
            attention_mask = to_numpy(attention_mask.float())
            position_ids =  to_numpy(position_ids)
            self.model.token_length = input_ids_shape

            token  = self.model.forward_first(inputs_embeds, position_ids, attention_mask)
            out_ids = [int(token)]
            token_len = input_ids_shape
            valid_position_ids = position_ids.max()

            while int(token) not in [self.ID_AU_END, self.ID_END
                                ] and token_len < self.model.SEQLEN:
                token_len += 1
                valid_position_ids = valid_position_ids + 1
                position_ids = np.array([[valid_position_ids]])
                attention_mask = torch.zeros(
                        (1, 1, 1, 599)).float()
                attention_mask[:, :, :, token_len-1:] = -.0
                token = self.model.forward_next(token, position_ids, attention_mask)
                out_ids.append(int(token))
                #print(out_ids)
                output_text = self.processor.batch_decode(
                    out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                #print(''.join(output_text))
            output_text = self.processor.batch_decode(
                out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            print(''.join(output_text))


def main(args):
    model = Qwen2Audio(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help='path to the config file')
    parser.add_argument('-d', '--devid', type=int, default=0, help='device ID to use')
    # yapf: enable
    args = parser.parse_args()
    main(args)
