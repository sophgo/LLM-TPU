import argparse
import time
from transformers import AutoTokenizer


class ChatGLM():
    def __init__(self, args):
        # preprocess parameters, such as prompt & tokenizer
        # parameters
        self.EOS = None
        self.SEQLEN = None
        self.input_str = ""
        self.system_prompt = ""
        self.history = []

        # devid
        self.devices = [int(d) for d in args.devid.split(",")]

        # load tokenizer
        print("Load " + args.tokenizer_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )

        # warm up
        self.tokenizer.decode([0])
        print("Done!")
        self.history = list()
        self.EOS = self.tokenizer.eos_token_id
        self.decode_mode = "diff"

        # load model
        self.load_model(args)
    
    def chat(self):
        """
        Start a chat session.
        """
        # check
        if not self.EOS:
            raise NotImplementedError("Forget to set End of Sentence Token Id(EOS)")
        if not self.SEQLEN:
            raise NotImplementedError("Forget to set End of Sentence Token Id")

        # Instruct
        print(
            """\n===========================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
==========================================================="""
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
            pre_word = self.tokenizer.decode([token], skip_special_tokens=True)
            word = self.tokenizer.decode([token, token], skip_special_tokens=True)[len(pre_word):]
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

        self.update_history()

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

    def load_model(self, args):
        import chat
        self.model = chat.ChatGLM()
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
        self.history = []

    def update_history(self):
        if self.model.token_length >= self.SEQLEN:
            print("... (reach the maximal length)", flush=True, end="")
            self.history = []
        else:
            self.history.append({"role": "assistant", "content": self.answer_cur})

    def encode_tokens(self):
        tokens = self.tokenizer.build_chat_input(self.input_str, history=self.history)['input_ids'].tolist()[0]
        self.history.append({"role": "user", "content": self.input_str})
        return tokens


def main(args):
    model = ChatGLM(args)
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
    args = parser.parse_args()
    main(args)
