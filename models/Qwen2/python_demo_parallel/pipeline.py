import argparse

import chat
import time
from transformers import AutoTokenizer


class Qwen2():
    def __init__(self, args):
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]

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
        while token != self.EOS and self.model.token_length < self.model.SEQLEN:
            word = self.tokenizer.decode(token, skip_special_tokens=True)
            self.answer_token += [token]
            print(word, flush=True, end="")
            tok_num += 1
            token = self.model.forward_next(token)
        self.answer_cur = self.tokenizer.decode(self.answer_token)
        
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


    ## For Web Demo
    def stream_predict(self, query):
        """
        Stream the prediction for the given query.
        """
        self.answer_cur = ""
        self.input_str = query
        tokens = self.encode_tokens()

        for answer_cur, history in self._generate_predictions(tokens):
            yield answer_cur, history


    def _generate_predictions(self, tokens):
        """
        Generate predictions for the given tokens.
        """
        # First token
        next_token = self.model.forward_first(tokens)
        output_tokens = [next_token]

        # Following tokens
        while True:
            next_token = self.model.forward_next()
            if next_token == self.EOS:
                break
            output_tokens += [next_token]
            self.answer_cur = self.tokenizer.decode(output_tokens)
            if self.model.token_length >= self.model.SEQLEN:
                self.update_history()
                yield self.answer_cur + "\n\n\nReached the maximum length; The history context has been cleared.", self.history
                break
            else:
                yield self.answer_cur, self.history

        self.update_history()


def main(args):
    model = Qwen2(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
    parser.add_argument('-t', '--tokenizer_path', type=str, default="../support/token_config", help='path to the tokenizer file')
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--enable_history', action='store_true', help="if set, enables storing of history memory")
    args = parser.parse_args()
    main(args)
