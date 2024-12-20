import os
import sys
import json
import time
import random
import argparse
import numpy as np
from transformers import AutoTokenizer

import chat

class Qwen:
    def __init__(self, args):
        # preprocess parameters, such as prompt & tokenizer
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]
        self.model_path = args.model_path

        # other parameters
        self.seq_length_list = [8192,7168,6144,5120,4096,3072,2048,1024]
        self.prefill_length_list = [8192,7168,6144,5120,4096,3072,2048,1024]

        # load tokenizer
        print("Load " + args.tokenizer_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )

        # warm up
        self.tokenizer.decode([0])
        self.EOS = self.tokenizer.eos_token_id

        self.model = chat.Qwen()
        self.init_params(args)

    def load_model(self, model_path, read_bmodel):
        load_start = time.time()
        self.model.init(self.devices, model_path, read_bmodel) # when read_bmodel = false, not to load weight, reuse weight
        load_end = time.time()
        print(f"\nLoad Time: {(load_end - load_start):.3f} s")

    def update_lora_and_lora_embedding(self, lora_path):
        net_idx = ",".join(str(2 * i) for i in range(28)) + ",56" # lora + lora_embedding
        mem_idx = ",".join(str(i) for i in range(28)) + ",28"
        weight_idx = ["1,2,3,4,5,6,9,10,13,14,16,17,19,20"]*28 + ["0,1"]

        start_time = time.time()
        self.model.update_bmodel_weight(self.model_path, lora_path, net_idx, mem_idx, weight_idx)
        end_time = time.time()
        print(f"\nLora Update Time: {(end_time - start_time):.3f} s")

    def empty_lora(self):
        net_idx = ",".join(str(2 * i) for i in range(28)) # lora
        mem_idx = ",".join(str(i) for i in range(28))
        weight_idx = ["1,2,3,4,5,6,9,10,13,14,16,17,19,20"]*28

        start_time = time.time()
        self.model.empty_bmodel_weight(self.model_path, net_idx, mem_idx, weight_idx)
        end_time = time.time()
        print(f"\nLora Empty Time: {(end_time - start_time):.3f} s")

    def empty_lora_embedding(self):
        net_idx = "56" # lora_embedding
        mem_idx = "28"
        weight_idx =  ["0,1"]

        start_time = time.time()
        self.model.empty_bmodel_weight(self.model_path, net_idx, mem_idx, weight_idx)
        end_time = time.time()
        print(f"\nLora Empty Time: {(end_time - start_time):.3f} s")

    def empty_lora_and_lora_embedding(self):
        net_idx = ",".join(str(2 * i) for i in range(28)) + ",56" # lora + lora_embedding
        mem_idx = ",".join(str(i) for i in range(28)) + ",28"
        weight_idx = ["1,2,3,4,5,6,9,10,13,14,16,17,19,20"]*28 + ["0,1"]

        start_time = time.time()
        self.model.empty_bmodel_weight(self.model_path, net_idx, mem_idx, weight_idx)
        end_time = time.time()
        print(f"\nLora Empty Time: {(end_time - start_time):.3f} s")

    def init_params(self, args):
        self.model.temperature = args.temperature
        self.model.top_p = args.top_p
        self.model.repeat_penalty = args.repeat_penalty
        self.model.repeat_last_n = args.repeat_last_n
        self.model.max_new_tokens = args.max_new_tokens
        self.model.generation_mode = args.generation_mode
        self.model.lib_path = args.lib_path
        self.model.embedding_path = args.embedding_path
        self.model.enable_lora_embedding = args.enable_lora_embedding

    def encode_tokens(self, prompt):
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer(text).input_ids
        return tokens

    def stream_answer(self, tokens, max_tok_num):
        """
        Stream the answer for the given tokens.
        """
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []

        print()
        # First token
        first_start = time.time()
        token = self.model.forward_first(tokens)
        first_end = time.time()
        # Following tokens
        while (max_tok_num > 0 and tok_num < max_tok_num) or (
            max_tok_num == 0
            and token != self.EOS
            and self.model.total_length < self.model.SEQLEN
        ):
            word = self.tokenizer.decode(token, skip_special_tokens=True)
            self.answer_token += [token]
            print(word, flush=True, end="")
            tok_num += 1
            token = self.model.forward_next()
        self.answer_cur = self.tokenizer.decode(self.answer_token)

        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration

        print()
        print(f"FTL Time: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

    def get_seq_index(self, total_length, in_length):
        seq_index = []
        for index, (t_length, i_length) in enumerate(zip(self.seq_length_list, self.prefill_length_list)):
            if t_length >= total_length and i_length >= in_length:
                seq_index.append(index)
        return seq_index

    def test_sample(self):
        sample_str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + "Give me a short introduction to large language model." + "<|im_end|>\n<|im_start|>assistant\n"

        # ===------------------------------------------------------------===
        # Model Init
        # ===------------------------------------------------------------===
        self.model.init_decrypt()
        self.model.prefill_reuse = 0
        self.model.stage_idx = 0
        self.load_model(self.model_path, read_bmodel=True)

        # sample 0
        in_tokens = self.tokenizer.encode(sample_str)

        in_length = len(in_tokens)
        out_length = 20
        total_length = in_length + out_length

        self.stream_answer(in_tokens, out_length)

        # ===------------------------------------------------------------===
        # Deinit
        # ===------------------------------------------------------------===
        self.model.deinit_decrypt()
        self.model.deinit()

    def test_lora(self, lora_path):
        sample_str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + "Give me a short introduction to large language model." + "<|im_end|>\n<|im_start|>assistant\n"

        # ===------------------------------------------------------------===
        # Model Init
        # ===------------------------------------------------------------===
        self.model.init_decrypt()
        self.model.prefill_reuse = 0
        self.model.stage_idx = 0
        self.load_model(self.model_path, read_bmodel=True)

        # sample 0
        in_tokens = self.tokenizer.encode(sample_str)

        in_length = len(in_tokens)
        out_length = 20
        total_length = in_length + out_length

        # load lora model
        self.empty_lora_and_lora_embedding()
        self.update_lora_and_lora_embedding(lora_path)
        self.stream_answer(in_tokens, out_length)

        # ===------------------------------------------------------------===
        # Deinit
        # ===------------------------------------------------------------===
        self.model.deinit_decrypt()
        self.model.deinit()

"""
-1: your input is empty or exceed the maximum length
-2: can not to create handle
-3: can not to create bmrt
-4: can not to load bmodel, maybe your key is wrong
-5: can not to inference bmodel
-6: addr_mode = 0, but must set addr_mode =1
"""
def main(args):
    dir_path = args.abnormal_path
    start_time = time.time()

    engine = Qwen(args)
    for idx in range(100_0000):
        print(f"---------------------------{idx}---------------------------")
        print("---------------------------(1) test embedding---------------------------")
        embedding_path_list = [
            "embedding.bin", "embedding.bin.empty", "embedding.bin.splitaa",
            "embedding.bin.splitab", "embedding.bin.split0", "embedding.bin.split1"
        ]
        random.shuffle(embedding_path_list)
        for embedding_path in embedding_path_list:
            try:
                engine.model.enable_lora_embedding = False
                engine.model.embedding_path = f"{dir_path}/{embedding_path}"
                engine.test_sample()
            except Exception as e:
                print(f"{type(e).__name__} : {str(e)}")
            finally:
                engine.model.deinit()


        engine.model.embedding_path = f"{dir_path}/{embedding_path_list[0]}"
        print("---------------------------(2) test lora---------------------------")
        lora_path_list = [
            "encrypted_lora_weights_r64.bin", "encrypted_lora_weights.bin.empty",
            "encrypted_lora_weights_r32.bin", "encrypted_lora_weights_r96.bin"
        ]
        random.shuffle(lora_path_list)
        for lora_path in lora_path_list:
            try:
                engine.model.enable_lora_embedding = False
                engine.test_lora(f"{dir_path}/{lora_path}")

                engine.model.enable_lora_embedding = True
                engine.test_lora(f"{dir_path}/{lora_path}")
            except Exception as e:
                print(f"{type(e).__name__} : {str(e)}")
            finally:
                engine.model.deinit()


    end_time = time.time()
    print(f"\nTotal Time: {(end_time - start_time):.3f} s")
    print("Status Code: ", engine.model.status_code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True, help="path to the bmodel")
    parser.add_argument('-t', '--tokenizer_path', type=str, default="../support/token_config", help='path to the tokenizer file')
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')
    parser.add_argument('--top_p', type=float, default=1.0, help='cumulative probability of token words to consider as a set of candidates')
    parser.add_argument('--repeat_penalty', type=float, default=1.2, help='penalty for repeated tokens')
    parser.add_argument('--repeat_last_n', type=int, default=32, help='repeat penalty for recent n tokens')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new token length to generate')
    parser.add_argument('--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
    parser.add_argument('--prompt_mode', type=str, choices=["prompted", "unprompted"], default="prompted", help='use prompt format or original input')
    parser.add_argument('--enable_history', action='store_true', help="if set, enables storing of history memory")
    parser.add_argument('--lib_path', type=str, default='', help='lib path by user')
    parser.add_argument('--abnormal_path', type=str, default='', help='abnormal path to test exception')
    parser.add_argument('--embedding_path', type=str, default='', help='binary embedding path')
    parser.add_argument('--lora_path', type=str, default='', help='binary lora path')
    parser.add_argument('--enable_lora_embedding', action='store_true', help="if set, enables lora embedding")
    args = parser.parse_args()
    main(args)
