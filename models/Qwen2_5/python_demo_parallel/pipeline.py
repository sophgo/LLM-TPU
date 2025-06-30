import argparse

import chat
import time
from transformers import AutoTokenizer


class Qwen2():
    def __init__(self, args):
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]

        # load tokenizer
        print("Load " + args.config_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.config_path, trust_remote_code=True
        )

        # warm up
        self.tokenizer.decode([0])

        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = "You are a helpful assistant."
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.EOS = self.tokenizer.eos_token_id
        self.enable_history = args.enable_history

        self.model = chat.Qwen()
        self.load_model(args.model_path)


    def load_model(self, model_path):
        load_start = time.time()
        self.model.init(self.devices, model_path)
        load_end = time.time()
        print(f"\nLoad Time: {(load_end - load_start):.3f} s")


    def clear(self):
        self.history = [{"role": "system", "content": self.system_prompt}]


    def update_history(self):
        if self.model.token_length >= self.model.SEQLEN:
            print("... (reach the maximal length)", flush=True, end="")
            self.history = [{"role": "system", "content": self.system_prompt}]
        else:
            self.history.append({"role": "assistant", "content": self.answer_cur})


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
        while token != self.EOS and self.model.token_length < self.model.SEQLEN:
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "�" in word:
                token = self.model.forward_next(token)
                tok_num += 1
                continue
            self.answer_token += full_word_tokens
            print(word, flush=True, end="")
            tok_num += 1
            full_word_tokens = []
            token = self.model.forward_next(token)

        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration

        if self.enable_history:
            self.answer_cur = self.tokenizer.decode(self.answer_token)
            self.update_history()
        else:
            self.clear()

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

def main(args):
    model = Qwen2(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="../config", help='path to the config file')
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--enable_history', action='store_true', help="if set, enables storing of history memory")
    args = parser.parse_args()
    main(args)
