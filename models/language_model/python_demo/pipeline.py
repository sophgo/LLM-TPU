import os
import json
import time
import random
import argparse
from transformers import AutoTokenizer

import sys
import chat

class Model:
    def __init__(self, args):
        # preprocess parameters, such as prompt & tokenizer
        self.devices = [int(d) for d in args.devid.split(",")]
        config_path = os.path.join(args.dir_path, "config.json")
        tokenizer_path = os.path.join(args.dir_path, "tokenizer")

        # load tokenizer
        print("Load " + tokenizer_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, use_fast=False
        )

        # config
        with open(config_path, 'r') as file:
            self.config = json.load(file)

        # warm up
        self.tokenizer.decode([0])

        # Initialize model-specific mapper dynamically
        self.model_type = args.model_type if args.model_type else self.config['model_type']
        self.map(self.model_type)

        # Initialize model
        self.model = chat.Model()
        self.init_params(args)
        self.load_model(args.model_path, read_bmodel=True)

    def map(self, model_type):
        """Abstract model-specific mapper into a dictionary."""
        if model_type == "qwen2":
            self.EOS = self.tokenizer.eos_token_id
            self.append_user = lambda history, input_str: history.append(
                {"role": "user", "content": input_str}
            )
            self.append_assistant = lambda history, answer_str: history.append(
                {"role": "assistant", "content": answer_str}
            )
            self.apply_chat_template = lambda history: self.tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
            self.history_init = [{"role": "system", "content": "You are a helpful assistant."}]
        elif model_type == "qwen":
            self.EOS = self.tokenizer.im_end_id
            self.append_user = lambda history, input_str: history.append(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(input_str)
            )
            self.append_assistant = lambda history, answer_str: history.append(answer_str)
            self.apply_chat_template = lambda history: "".join(history)
            self.history_init = ["<|im_start|>system\nYou are a helpful assistant."]
        elif model_type == "llama":
            system_prompt = "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
                            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
                            "Please ensure that your responses are socially unbiased and positive in nature. " \
                            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
                            "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
            self.EOS = self.tokenizer.eos_token_id
            self.append_user = lambda history, input_str: history.append(
                "{} [/INST] ".format(input_str)
            )
            self.append_assistant = lambda history, answer_str: history.append(
                "{} </s><s>[INST] ".format(answer_str)
            )
            self.apply_chat_template = lambda history: "".join(history)
            self.history_init = [system_prompt]
            self.tokenizer.add_prefix_space = False
        elif model_type == "lwm":
            system_prompt = "You are a helpful assistant. "
            self.EOS = self.tokenizer.eos_token_id
            self.append_user = lambda history, input_str: history.append(
                "USER: {} ASSISTANT: ".format(input_str)
            )
            self.append_assistant = lambda history, answer_str: history.append(
                "{} ".format(answer_str)
            )
            self.apply_chat_template = lambda history: "".join(history)
            self.history_init = [system_prompt]
            self.tokenizer.add_prefix_space = False
        else:
            raise NotImplementedError(f"{model_type} not support now")
        return

    def load_model(self, model_path, read_bmodel):
        load_start = time.time()
        self.model.init(self.devices, model_path, read_bmodel) # when read_bmodel = false, not to load weight, reuse weight
        load_end = time.time()
        print(f"\nLoad Time: {(load_end - load_start):.3f} s")

    def init_params(self, args):
        self.model.temperature = args.temperature
        self.model.top_p = args.top_p
        self.model.repeat_penalty = args.repeat_penalty
        self.model.repeat_last_n = args.repeat_last_n
        self.model.max_new_tokens = args.max_new_tokens
        self.model.generation_mode = args.generation_mode
        self.model.embedding_path = os.path.join(args.dir_path, "embedding.bin")
        self.model.NUM_LAYERS = self.config["num_hidden_layers"]

        self.enable_history = args.enable_history
        self.init_history()

    def init_history(self):
        self.history = self.history_init

    def update_history(self):
        if self.model.total_length >= self.model.SEQLEN:
            print("... (reach the maximal length)", flush=True, end="")
            self.init_history()
        else:
            self.append_assistant(self.history, self.answer_cur)

    def encode_tokens(self):
        self.append_user(self.history, self.input_str)
        text = self.apply_chat_template(self.history)
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
                self.init_history()
            # Chat
            else:
                tokens = self.encode_tokens()

                # check tokens
                if not tokens:
                    print("Sorry: your question is empty!!")
                    return
                if len(tokens) > self.model.SEQLEN:
                    print(
                        "The maximum question length should be shorter than {} but we get {} instead.".format(
                            self.model.SEQLEN, len(tokens)
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

        # First token
        first_start = time.time()
        token = self.model.forward_first(tokens)
        first_end = time.time()
        # Following tokens
        full_word_tokens = []
        while token != self.EOS and self.model.total_length < self.model.SEQLEN:
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "�" in word:
                token = self.model.forward_next()
                tok_num += 1
                continue
            self.answer_token += full_word_tokens
            print(word, flush=True, end="")
            tok_num += 1
            full_word_tokens = []
            token = self.model.forward_next()

        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration

        if self.enable_history:
            self.answer_cur = self.tokenizer.decode(self.answer_token)
            self.update_history()
        else:
            self.init_history()

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

def main(args):
    model = Model(args)
    model.chat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--dir_path", type=str, default="./tmp", help="dir path to the config/embedding/tokenizer")
    parser.add_argument('-b', '--model_path', type=str, required=True, help='path to the bmodel file')
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')
    parser.add_argument('--top_p', type=float, default=1.0, help='cumulative probability of token words to consider as a set of candidates')
    parser.add_argument('--repeat_penalty', type=float, default=1.2, help='penalty for repeated tokens')
    parser.add_argument('--repeat_last_n', type=int, default=32, help='repeat penalty for recent n tokens')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new token length to generate')
    parser.add_argument('--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
    parser.add_argument('--enable_history', action='store_true', help="if set, enables storing of history memory")
    parser.add_argument('--model_type', type=str, help="model type")
    args = parser.parse_args()
    main(args)
