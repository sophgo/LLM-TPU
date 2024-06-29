import argparse

import chat
import json
import time
from transformers import AutoTokenizer


class Qwen1_5():
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
        self.SEQLEN = self.model.SEQLEN
        self.MAX_SHARE_LENGTH = self.model.MAX_SHARE_LENGTH


    def clear(self):
        self.history = [{"role": "system", "content": self.system_prompt}]


    def encode_tokens(self):
        self.history.append({"role": "user", "content": self.input_str})
        text = self.tokenizer.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer(text).input_ids
        return tokens


    def chat(self):
        """
        Start a chat session.
        """
        # Instruct
        print(
            """\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
================================================================="""
        )
        # Stop Chatting with "exit" input
        while True:
            self.input_str = input("\nQuestion: ")
            # Quit
            if self.input_str in ["exit", "q", "quit"]:
                break
            # New Chat
            elif self.input_str in ["clear", "new"]:
                self.clear()
            # Chat
            else:
                tokens = self.encode_tokens()

                # check tokens
                if not tokens:
                    print("Sorry: your question is empty!!")
                    return
                if len(tokens) > self.SEQLEN:
                    print(
                        "The maximum question length should be shorter than {} but we get {} instead.".format(
                            self.SEQLEN, len(tokens)
                        )
                    )
                    return

                print("\nAnswer: ", end="")
                self.stream_answer(tokens)


    def stream_answer(self, tokens):
        """
        Stream the answer for the given tokens.
        """
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []

        print()
        # First token
        first_start = time.time()
        token = self.model.forward_unshare(tokens)
        first_end = time.time()
        # Following tokens
        while token != self.EOS and self.model.unshare_length < self.SEQLEN + self.model.MAX_SHARE_LENGTH:
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
        print(f"Unshare FTL Time: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")


    def read_json(self, json_path, task_id):
        with open(json_path, 'r') as file:
            text = json.load(file)
        system_str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        content_str = system_str + text[task_id]['content']
        question_str = text[task_id]['question'] + "<|im_end|>\n<|im_start|>assistant\n"
        return content_str, question_str

    def test_share_cache(self):
        share_str, unshare_str_0 = self.read_json("sophgo_kv_cache_share_test_case.json", 0)
        _, unshare_str_1 = self.read_json("sophgo_kv_cache_share_test_case.json", 1)
        _, unshare_str_2 = self.read_json("sophgo_kv_cache_share_test_case.json", 2)
        # share_str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        # unshare_str_0 = "can you help me<|im_end|>\n<|im_start|>assistant\n"
        # unshare_str_1 = "tell me a love story<|im_end|>\n<|im_start|>assistant\n"

        #===------------------------------------------------------------===
        # Model 0
        #===------------------------------------------------------------===
        self.model.encrypt_bmodel(self.model_list[0])

        # load model 0
        self.load_model(self.model_list[0])
        self.model.empty_kvcache()

        # share prefill
        share_start = time.time()
        share_tokens = self.tokenizer.encode(share_str)
        self.model.forward_first(share_tokens)
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
        self.model.encrypt_bmodel(self.model_list[1])
        self.load_model(self.model_list[1])

        # share prefill
        share_start = time.time()
        share_tokens = self.tokenizer.encode(share_str)
        # self.model.forward_first(share_tokens)
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


def main(args):
    model = Qwen1_5(args)
    model.test_share_cache()

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
