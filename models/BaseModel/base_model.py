import time
from transformers import AutoTokenizer


class BaseModel:
    def __init__(self, args):
        # preprocess parameters
        self.input_str = ""
        self.system_prompt = ""

        # model parameters
        self.token_length = 0
        self.SEQLEN = None

        # devid
        self.devices = [int(d) for d in args.devid.split(",")]

        # load tokenizer
        print("Load " + args.tokenizer_path + " ...")
        self.sp = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )

        # warm up
        self.sp.decode([0])
        print("Done!")

    def chat(self):
        # Instruct:
        print(
            f"""\n===========================================================
1. If you want to quit, please entry one of [q, quit, exit]
2. To create new chat-session, please entry one of [clear, new]
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

                print("\nAnswer: ", end="")
                self.stream_answer(tokens)

    # stat time cost
    def stream_answer(self, tokens):
        tok_num = 0
        self.answer_cur = ""

        if not tokens:
            print("Sorry: your question is too wierd!!")
            return
        if self.token_length > self.SEQLEN:
            print(
                "The maximum question length should be shorter than {} but we get {} instead.".format(
                    self.SEQLEN, self.token_length
                )
            )
            return

        # First token
        pre_token = 0
        first_start = time.time()
        token = self.forward_first(tokens)
        first_end = time.time()

        # Following tokens
        while token != self.sp.eos_token_id and self.token_length < self.SEQLEN:
            word = self.decode_tokens(pre_token, token)
            pre_token = token
            print(word, flush=True, end="")
            if self.token_length < self.SEQLEN:
                self.token_length += 1
            tok_num += 1
            token = self.forward_next()

        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration

        self.update_history()

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

    def stream_predict(self, query, messages=None):
        self.answer_cur = ""

        self.input_str = query
        tokens = self.generate_tokens()

        if not tokens:
            print("Sorry: your question is too wierd!!")
            return
        if self.token_length > self.SEQLEN:
            print(
                "The maximum question length should be shorter than {} but we get {} instead.".format(
                    self.SEQLEN, self.token_length
                )
            )
            return

        # First token
        next_token = self.forward_first(tokens)
        output_tokens = [next_token]

        # Following tokens
        while True:
            next_token = self.forward_next(next_token)
            if next_token == self.EOS:
                break
            if self.token_length < self.SEQLEN:
                self.token_length += 1
            output_tokens += [next_token]
            self.answer_cur = self.sp.decode(output_tokens)
            self.history_update()
            yield self.answer_cur, self.messages

    def forward_first(self, tokens):
        token = self.model.forward_first(tokens)
        return token

    def forward_next(self):
        token = self.model.forward_next()
        return token

    def encode_tokens(self):
        pass

    def decode_tokens(self, pre_token, token):
        if self.decode_mode == "diff":
            pre_word = self.sp.decode([pre_token], skip_special_tokens=True)
            word = self.sp.decode([pre_token, token], skip_special_tokens=True)
            return word[len(pre_word):]
        else:
            word = self.sp.decode([token], skip_special_tokens=True)
            return word

    def load_model(self):
        pass

    def clear(self):
        pass

    def history_update(self):
        pass
