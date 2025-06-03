import argparse

import time
from transformers import AutoTokenizer


class Qwen():
    def __init__(self, args):
        # preprocess parameters, such as prompt & tokenizer
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]

        # load tokenizer
        print("Load " + args.tokenizer_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )

        # warm up
        self.tokenizer.decode([0])

        self.system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        self.prompt = (
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.EOS = self.tokenizer.im_end_id # tokenizer.encode("<|im_end|>")
        self.history = [self.system_prompt]
        self.enable_history = args.enable_history

        # load model
        self.load_model(args)


    def load_model(self, args):
        if args.decode_mode == "jacobi":
            import chat_jacobi
            self.model = chat_jacobi.Qwen()
        elif args.decode_mode == "basic":
            import chat
            self.model = chat.Qwen()
            self.model.init(self.devices, args.model_path)
            self.model.temperature = args.temperature
            self.model.top_p = args.top_p
            self.model.repeat_penalty = args.repeat_penalty
            self.model.repeat_last_n = args.repeat_last_n
            self.model.max_new_tokens = args.max_new_tokens
            self.model.generation_mode = args.generation_mode
            self.model.prompt_mode = args.prompt_mode
        self.SEQLEN = self.model.SEQLEN


    def clear(self):
        self.history = [self.system_prompt]


    def update_history(self):
        if self.model.token_length >= self.SEQLEN:
            print("... (reach the maximal length)", flush=True, end='')
            self.history = [self.system_prompt]
        else:
            self.history[-1] = self.history[-1] + self.answer_cur


    def encode_tokens(self):
        self.history.append(self.prompt.format(self.input_str))
        text = "".join(self.history)
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

        # First token
        first_start = time.time()
        token = self.model.forward_first(tokens)
        first_end = time.time()
        # Following tokens
        while token != self.EOS and self.model.token_length < self.SEQLEN:
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
            if self.model.token_length >= self.SEQLEN:
                self.update_history()
                yield self.answer_cur + "\n\n\nReached the maximum length; The history context has been cleared.", self.history
                break
            else:
                yield self.answer_cur, self.history

        self.update_history()


def main(args):
    model = Qwen(args)
    model.chat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
    parser.add_argument('-t', '--tokenizer_path', type=str, default="../support/token_config", help='path to the tokenizer file')
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')
    parser.add_argument('--top_p', type=float, default=1.0, help='cumulative probability of token words to consider as a set of candidates')
    parser.add_argument('--repeat_penalty', type=float, default=1.0, help='penalty for repeated tokens')
    parser.add_argument('--repeat_last_n', type=int, default=32, help='repeat penalty for recent n tokens')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new token length to generate')
    parser.add_argument('--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
    parser.add_argument('--prompt_mode', type=str, choices=["prompted", "unprompted"], default="prompted", help='use prompt format or original input')
    parser.add_argument('--decode_mode', type=str, default="basic", choices=["basic", "jacobi"], help='mode for decoding')
    parser.add_argument('--enable_history', action='store_true', help="if set, enables storing of history memory.")
    args = parser.parse_args()
    main(args)
