# ==============================================================================
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import argparse

import chat
import time
from transformers import AutoTokenizer, GenerationConfig


class Qwen3():

    def __init__(self, args):
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]

        # load tokenizer
        print("Load " + args.config_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.config_path, trust_remote_code=True)

        # warm up
        self.tokenizer.decode([0])

        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = "You are a helpful assistant."
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.EOS = [self.tokenizer.eos_token_id]
        self.enable_history = args.enable_history

        self.model = chat.Qwen()
        self.init_params(args)
        self.load_model(args.model_path)
        if self.model.support_prefill_kv:
            print("Model supports prefill kv, using prefill mode.")
            self.enable_history = True

    def load_model(self, model_path):
        load_start = time.time()
        self.model.init(self.devices, model_path)
        load_end = time.time()
        print(f"\nLoad Time: {(load_end - load_start):.3f} s")

    def init_params(self, args):
        self.model.generation_mode = "greedy"
        self.stop_strings = []
        if args.do_sample:
            gen_config = GenerationConfig.from_pretrained(args.config_path)
            self.model.generation_mode = "sample"
            self.model.temperature = gen_config.temperature
            self.model.top_p = gen_config.top_p
            self.model.top_k = gen_config.top_k
            self.model.penalty = gen_config.repetition_penalty
            if gen_config.eos_token_id is not None:
                if isinstance(gen_config.eos_token_id, int):
                    self.EOS.append(gen_config.eos_token_id)
                if isinstance(gen_config.eos_token_id, list):
                    self.EOS.extend(gen_config.eos_token_id)
            if gen_config.stop_strings is not None:
                self.stop_strings = gen_config.stop_strings

    def clear(self):
        if self.model.support_prefill_kv:
            self.model.clear_kv()
        self.history = [{"role": "system", "content": self.system_prompt}]

    def update_history(self):
        if self.model.history_length >= self.model.SEQLEN:
            print("... (reach the maximal length)", flush=True, end="")
            self.clear()
        elif self.model.support_prefill_kv:
            self.history.clear()
        else:
            self.history.append({"role": "assistant", "content": self.answer_cur})

    def encode_tokens(self):
        self.history.append({"role": "user", "content": self.input_str})
        text = self.tokenizer.apply_chat_template(self.history,
                                                  tokenize=False,
                                                  add_generation_prompt=True)
        tokens = self.tokenizer(text).input_ids
        return tokens

    def chat(self):
        """
        Start a chat session.
        """
        # Instruct
        print("""\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
=================================================================""")
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
                if len(tokens) > self.model.MAX_INPUT_LENGTH:
                    print(
                        "The maximum question length should be shorter than {} but we get {} instead."
                        .format(self.model.MAX_INPUT_LENGTH, len(tokens)))
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
        while token not in self.EOS and self.model.history_length < self.model.SEQLEN:
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "�" in word:
                token = self.model.forward_next()
                tok_num += 1
                continue
            self.answer_token += full_word_tokens
            self.answer_cur += word
            if any(self.answer_cur.endswith(stop) for stop in self.stop_strings):
                break
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
            self.update_history()
        else:
            self.clear()

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")


def main(args):
    model = Qwen3(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="../config", help='path to the tokenizer file')
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--do_sample', action='store_true', help="if set, generate tokens by sample parameters")
    parser.add_argument('--enable_history', action='store_true', help="if set, enables storing of history memory")
    # yapf: enable
    args = parser.parse_args()
    main(args)
