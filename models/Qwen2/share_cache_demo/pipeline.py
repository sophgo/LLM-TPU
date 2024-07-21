import argparse

import chat
import json
import time
from transformers import AutoTokenizer


class Qwen():
    def __init__(self, args):
        # preprocess parameters, such as prompt & tokenizer
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]
        self.model_list = [d for d in args.model_path_list.split(",")]

        # load tokenizer
        print("Load " + args.tokenizer_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )

        # warm up
        self.tokenizer.decode([0])

        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = "You are a helpful assistant."
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.EOS = self.tokenizer.eos_token_id

        self.model = chat.Qwen()
        self.model.memory_prealloc = args.memory_prealloc
        self.model.is_decrypt = args.is_decrypt
        self.init_params(args)


    def load_model(self, model_path):
        load_start = time.time()
        self.model.init(self.devices, model_path)
        load_end = time.time()
        print(f"\nLoad Time: {(load_end - load_start):.3f} s")


    def init_params(self, args):
        self.model.temperature = args.temperature
        self.model.top_p = args.top_p
        self.model.repeat_penalty = args.repeat_penalty
        self.model.repeat_last_n = args.repeat_last_n
        self.model.max_new_tokens = args.max_new_tokens
        self.model.generation_mode = args.generation_mode


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
        elif inference_mode == "share":
            token = self.model.forward_unshare(tokens)
        else:
            raise ValueError(f"Not support {inference_mode}")
        first_end = time.time()
        # Following tokens
        # while token != self.EOS and self.model.total_length < self.model.SEQLEN:
        while tok_num < max_tok_num:
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
        elif inference_mode == "share":
            print(f"Unshare FTL Time: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")


    def read_json(self, json_path, task_id):
        with open(json_path, 'r') as file:
            text = json.load(file)
        system_str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"

        content_str = ""
        if "content" in text[task_id]:
            content_str = system_str + text[task_id]['content']
        question_str = text[task_id]['question'] + "<|im_end|>\n<|im_start|>assistant\n"
        return content_str, question_str


    def test_share_cache(self):
        json_path = "../../../assets/sophgo_kv_cache_share_test_case.json"
        share_str, unshare_str_0 = self.read_json(json_path, 0)
        _, unshare_str_1 = self.read_json(json_path, 1)
        _, unshare_str_2 = self.read_json(json_path, 2)
        #share_str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        #unshare_str_0 = "can you help me<|im_end|>\n<|im_start|>assistant\n"
        #unshare_str_1 = "tell me a love story<|im_end|>\n<|im_start|>assistant\n"
        #unshare_str_2 = "tell me a love story<|im_end|>\n<|im_start|>assistant\n"

        #===------------------------------------------------------------===
        # Model 0
        #===------------------------------------------------------------===
        if self.model.is_decrypt:        
            self.model.encrypt_bmodel(self.model_list[0])

        # load model 0
        self.model.io_alone_reuse = False
        self.load_model(self.model_list[0])
        # self.model.empty_kvcache()

        # share prefill
        share_start = time.time()
        share_tokens = self.tokenizer.encode(share_str)
        self.model.forward_share(share_tokens)
        share_end = time.time()
        print(f"\nShare FTL Time: {(share_end - share_start):.3f} s")

        # task 0
        unshare_tokens_0 = self.tokenizer.encode(unshare_str_0)
        self.stream_answer(unshare_tokens_0)

        # task 1
        unshare_tokens_1 = self.tokenizer.encode(unshare_str_1)
        self.stream_answer(unshare_tokens_1)

        # task 2
        unshare_tokens_2 = self.tokenizer.encode(unshare_str_2)
        self.stream_answer(unshare_tokens_2)

        # free memory
        self.model.free_device()

        #===------------------------------------------------------------===
        # Model 1
        #===------------------------------------------------------------===
        # load model 1
        self.model.io_alone_reuse = True
        if self.model.is_decrypt:
            self.model.encrypt_bmodel(self.model_list[1])
        self.load_model(self.model_list[1])

        # share prefill
        share_start = time.time()
        # share_tokens = self.tokenizer.encode(share_str)
        # self.model.forward_share(share_tokens)
        share_end = time.time()
        print(f"\nShare FTL Time: {(share_end - share_start):.3f} s")

        # task 0
        unshare_tokens_0 = self.tokenizer.encode(unshare_str_0)
        self.stream_answer(unshare_tokens_0)

        # task 1
        unshare_tokens_1 = self.tokenizer.encode(unshare_str_1)
        self.stream_answer(unshare_tokens_1)

        # task 2
        unshare_tokens_2 = self.tokenizer.encode(unshare_str_2)
        self.stream_answer(unshare_tokens_2)

        #===------------------------------------------------------------===
        # Deinit
        #===------------------------------------------------------------===
        self.model.deinit()


    def test_share_cache_1(self):
        json_path = "../../../assets/long_case.json"
        share_str, unshare_str_0 = self.read_json(json_path, 0)
        _, unshare_str_1 = self.read_json(json_path, 1)
        _, unshare_str_2 = self.read_json(json_path, 2)
        #share_str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        #unshare_str_0 = "can you help me<|im_end|>\n<|im_start|>assistant\n"
        #unshare_str_1 = "tell me a love story<|im_end|>\n<|im_start|>assistant\n"
        #unshare_str_2 = "tell me a love story<|im_end|>\n<|im_start|>assistant\n"

        #===------------------------------------------------------------===
        # Model 0
        #===------------------------------------------------------------===
        if self.model.is_decrypt:        
            self.model.encrypt_bmodel(self.model_list[0])

        # load model 0
        # load sophgo_V4.6016s.1600us.8192seq.8192max.1dev_dyn.bmodel
        self.model.io_alone_reuse = False
        self.load_model(self.model_list[0])
        # self.model.empty_kvcache()

        # share prefill
        share_start = time.time()
        share_tokens = self.tokenizer.encode(
            share_str,
            max_length=6016,
            truncation=True,
            padding='max_length'
        )

        self.model.forward_share(share_tokens)
        share_end = time.time()
        print(f"\nShare FTL Time: {(share_end - share_start):.3f} s")

        # task 15
        unshare_tokens = self.tokenizer.encode(unshare_str_0)
        self.stream_answer(unshare_tokens, "share", 422)

        # task 16
        unshare_tokens = self.tokenizer.encode(unshare_str_1)
        self.stream_answer(unshare_tokens, "share", 438)

        # free memory
        self.model.free_device()

        #===------------------------------------------------------------===
        # Model 1
        #===------------------------------------------------------------===
        # load model 1
        # sophgo_V4.6016s.1024us.7552seq.8192max.1dev_dyn.bmodel
        self.model.io_alone_reuse = True
        if self.model.is_decrypt:
            self.model.encrypt_bmodel(self.model_list[1])
        self.load_model(self.model_list[1])

        # share prefill
        share_start = time.time()
        # share_tokens = self.tokenizer.encode(share_str)
        # self.model.forward_share(share_tokens)
        share_end = time.time()
        print(f"\nShare FTL Time: {(share_end - share_start):.3f} s")

        # task 3
        unshare_tokens = self.tokenizer.encode(unshare_str_0)
        self.stream_answer(unshare_tokens, "share", 139)

        # task 5
        unshare_tokens = self.tokenizer.encode(unshare_str_1)
        self.stream_answer(unshare_tokens, "share", 160)

        # task 12
        unshare_tokens = self.tokenizer.encode(unshare_str_2)
        self.stream_answer(unshare_tokens, "share", 281)

        # task 13
        unshare_tokens = self.tokenizer.encode(unshare_str_0)
        self.stream_answer(unshare_tokens, "share", 281)
        
        # task 9
        unshare_tokens = self.tokenizer.encode(unshare_str_1)
        self.stream_answer(unshare_tokens, "share", 755)

        # task 11
        unshare_tokens = self.tokenizer.encode(unshare_str_2)
        self.stream_answer(unshare_tokens, "share", 713)

        # task 4
        unshare_tokens = self.tokenizer.encode(unshare_str_0)
        self.stream_answer(unshare_tokens, "share", 322)

        # task 10
        unshare_tokens = self.tokenizer.encode(unshare_str_1)
        self.stream_answer(unshare_tokens, "share", 441)


        # share prefill
        share_start = time.time()
        share_tokens = self.tokenizer.encode(
            share_str,
            max_length=6000,
            truncation=True,
            padding='max_length'
        )

        self.model.forward_share(share_tokens)
        share_end = time.time()
        print(f"\nShare FTL Time: {(share_end - share_start):.3f} s")


        # task 1
        unshare_tokens = self.tokenizer.encode(unshare_str_0)
        self.stream_answer(unshare_tokens, "share", 106)

        # task 2
        unshare_tokens = self.tokenizer.encode(unshare_str_1)
        self.stream_answer(unshare_tokens, "share", 184)

        # free memory
        self.model.free_device()


        #===------------------------------------------------------------===
        # Model 2
        #===------------------------------------------------------------===
        # load model 2
        self.model.io_alone_reuse = False
        if self.model.is_decrypt:
            self.model.encrypt_bmodel(self.model_list[2])
        self.load_model(self.model_list[2])

        # task 6
        for _ in range(6):
            share_tokens = self.tokenizer.encode(
                share_str,
                max_length=908,
                truncation=True,
                padding='max_length'
            )
            unshare_tokens = self.tokenizer.encode(unshare_str_0)
            self.stream_answer(share_tokens + unshare_tokens, "normal", 7)

        # task 7
        for _ in range(5):
            share_tokens = self.tokenizer.encode(
                share_str,
                max_length=898,
                truncation=True,
                padding='max_length'
            )
            unshare_tokens = self.tokenizer.encode(unshare_str_1)
            self.stream_answer(share_tokens + unshare_tokens, "normal", 324)

        # task 8
        for _ in range(5):
            share_tokens = self.tokenizer.encode(
                share_str,
                max_length=162,
                truncation=True,
                padding='max_length'
            )
            unshare_tokens = self.tokenizer.encode(unshare_str_2)
            self.stream_answer(share_tokens + unshare_tokens, "normal", 106)

        # task 14
        for _ in range(1):
            share_tokens = self.tokenizer.encode(
                share_str,
                max_length=725,
                truncation=True,
                padding='max_length'
            )
            unshare_tokens = self.tokenizer.encode(unshare_str_0)
            self.stream_answer(share_tokens + unshare_tokens, "normal", 101)

        #===------------------------------------------------------------===
        # Deinit
        #===------------------------------------------------------------===
        self.model.deinit()


def main(args):
    start_time = time.time()
    model = Qwen(args)
    model.test_share_cache_1()
    end_time = time.time()

    print(f"\nTotal Time: {(end_time - start_time):.3f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path_list', type=str, required=True, help='path to the bmodel files')
    parser.add_argument('-t', '--tokenizer_path', type=str, default="../support/token_config", help='path to the tokenizer file')
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')
    parser.add_argument('--top_p', type=float, default=1.0, help='cumulative probability of token words to consider as a set of candidates')
    parser.add_argument('--repeat_penalty', type=float, default=1.2, help='penalty for repeated tokens')
    parser.add_argument('--repeat_last_n', type=int, default=32, help='repeat penalty for recent n tokens')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new token length to generate')
    parser.add_argument('--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
    parser.add_argument('--prompt_mode', type=str, choices=["prompted", "unprompted"], default="prompted", help='use prompt format or original input')
    parser.add_argument('--memory_prealloc', action='store_true', help="if set, prealloc weight memory for weight reuse")
    parser.add_argument('--is_decrypt', action='store_true', help="if set, will to decrypt bmodel before load")
    args = parser.parse_args()
    main(args)
