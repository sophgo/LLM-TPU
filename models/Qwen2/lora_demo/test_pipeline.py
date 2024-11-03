import os
import json
import time
import random
import argparse
from transformers import AutoTokenizer

import sys
import chat


class Qwen:
    def __init__(self, args):
        # preprocess parameters, such as prompt & tokenizer
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]
        self.model_path = args.model_path

        # other parameters
        self.seq_length_list = [10240,8192,7168,6144,5120,4096,3072,2048,1024]
        self.prefill_length_list = [8320,8192,7168,6144,5120,4096,3072,2048,1024]
        self.lora_path = args.lora_path
        self.zero_lora_path = args.zero_lora_path
        self.net_idx = ",".join(str(2 * i) for i in range(28)) + ",56" # lora + lora_embedding
        self.mem_idx = ",".join(str(i) for i in range(28)) + ",28"
        self.weight_idx = ["1,2,3,4,5,6,9,10,13,14,16,17,19,20"]*28 + ["0,1"]


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

    def update_bmodel(self, lora_path):
        start_time = time.time()
        self.model.update_bmodel_weight(self.model_path, lora_path, self.net_idx, self.mem_idx, self.weight_idx)
        end_time = time.time()
        print(f"\nLora Update Time: {(end_time - start_time):.3f} s")

    def empty_bmodel(self):
        start_time = time.time()
        self.model.empty_bmodel_weight(self.model_path, self.net_idx, self.mem_idx, self.weight_idx)
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

    def stream_answer(self, tokens, inference_mode, max_tok_num):
        """
        Stream the answer for the given tokens.
        """
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []

        print()
        # First token
        first_start = time.time()
        if inference_mode == "normal":
            token = self.model.forward_first(tokens)
        else:
            raise ValueError(f"Not support {inference_mode}")
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
        if inference_mode == "normal":
            print(f"FTL Time: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

    def read_json(self, json_path, task_id):
        with open(json_path, "r") as file:
            text = json.load(file)
        system_str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"

        content_str = ""
        if "content" in text[task_id]:
            content_str = system_str + text[task_id]["content"]
        question_str = text[task_id]["question"] + "<|im_end|>\n<|im_start|>assistant\n"
        return content_str, question_str

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

        self.stream_answer(in_tokens, "normal", out_length)

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
        self.update_bmodel(lora_path)
        self.stream_answer(in_tokens, "normal", out_length)

        # ===------------------------------------------------------------===
        # Deinit
        # ===------------------------------------------------------------===
        self.model.deinit_decrypt()
        self.model.deinit()

    def test_empty_lora(self):
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
        self.update_bmodel(lora_path)
        self.empty_bmodel()
        self.stream_answer(in_tokens, "normal", out_length)

        # ===------------------------------------------------------------===
        # Deinit
        # ===------------------------------------------------------------===
        self.model.deinit_decrypt()
        self.model.deinit()

    def test_zero_lora(self, lora_path):
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
        self.update_bmodel(lora_path)
        self.stream_answer(in_tokens, "normal", out_length)

        # ===------------------------------------------------------------===
        # Deinit
        # ===------------------------------------------------------------===
        self.model.deinit_decrypt()
        self.model.deinit()

    def test_empty_lora_with_loop(self, lora_path, loop_num):
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
        for _ in range(loop_num):
            self.update_bmodel(lora_path)
            self.empty_bmodel()
            self.stream_answer(in_tokens, "normal", out_length)

        for _ in range(loop_num):
            self.update_bmodel(lora_path)
            self.empty_bmodel()
        self.stream_answer(in_tokens, "normal", out_length)

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
    loop_num = 5
    dir_path = "test_lora"
    start_time = time.time()

    try:
        engine = Qwen(args)

        print("---------------------------(1)---------------------------")
        engine.enable_lora_embedding = False
        engine.test_sample()

        print("---------------------------(2)---------------------------")
        engine.enable_lora_embedding = False
        engine.test_lora(f"{dir_path}/encrypted_lora_weights_1_0.bin")
        engine.enable_lora_embedding = True
        engine.test_lora(f"{dir_path}/encrypted_lora_weights_0_1.bin")

        print("---------------------------(3)---------------------------")
        engine.enable_lora_embedding = False
        engine.test_empty_lora(f"{dir_path}/encrypted_lora_weights_1_0.bin")
        engine.test_empty_lora_with_loop(f"{dir_path}/encrypted_lora_weights_1_0.bin", loop_num)
        engine.enable_lora_embedding = True
        engine.test_empty_lora(f"{dir_path}/encrypted_lora_weights_0_1.bin")
        engine.test_empty_lora_with_loop(f"{dir_path}/encrypted_lora_weights_0_1.bin", loop_num)

        print("---------------------------(4)---------------------------")
        engine.test_zero_lora(f"{dir_path}/encrypted_lora_weights_0_0.bin")

        print("---------------------------(6)(8)---------------------------")
        engine.enable_lora_embedding = False
        engine.test_lora(f"{dir_path}/encrypted_lora_weights_10_0.bin")
        engine.test_lora(f"{dir_path}/encrypted_lora_weights_20_0.bin")
        engine.test_lora(f"{dir_path}/encrypted_lora_weights_30_0.bin")
        engine.enable_lora_embedding = True
        engine.test_lora(f"{dir_path}/encrypted_lora_weights_0_10.bin")
        engine.test_lora(f"{dir_path}/encrypted_lora_weights_0_20.bin")
        engine.test_lora(f"{dir_path}/encrypted_lora_weights_0_30.bin")

        print("All Right!")
    except RuntimeError:
        print("RuntimeError")
    except ValueError:
        print("ValueError")
    except:
        print("Error")

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
    parser.add_argument('--embedding_path', type=str, default='', help='binary embedding path')
    parser.add_argument('--lora_path', type=str, default='', help='binary lora path')
    parser.add_argument('--zero_lora_path', type=str, default='', help='zero binary lora path')
    parser.add_argument('--enable_lora_embedding', action='store_true', help="if set, enables lora embedding")
    args = parser.parse_args()
    main(args)
