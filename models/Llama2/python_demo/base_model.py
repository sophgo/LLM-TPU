import time
from transformers import AutoTokenizer


class BaseModel:
    def __init__(self, args):
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
        token = self.forward_first(tokens)
        first_end = time.time()
        # Following tokens
        while token != self.EOS and self.model.token_length < self.SEQLEN:
            pre_word = self.decode_tokens([token])
            word = self.decode_tokens([token, token])[len(pre_word):]
            self.answer_token += [token]
            print(word, flush=True, end="")
            tok_num += 1
            token = self.forward_next()
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
        next_token = self.forward_first(tokens)
        output_tokens = [next_token]

        # Following tokens
        while True:
            next_token = self.forward_next()
            if next_token == self.EOS:
                break
            output_tokens += [next_token]
            self.answer_cur = self.tokenizer.decode(output_tokens)
            if self.enable_history:
                self.update_history()
            else:
                self.clear()
            yield self.answer_cur, self.history

    def forward_first(self, tokens):
        """
        Forward the first token.
        """
        token = self.model.forward_first(tokens)
        return token

    def forward_next(self):
        """
        Forward the next token.
        """
        token = self.model.forward_next()
        return token

    def decode_tokens(self, token):
        """
        Decode the given token.
        """
        word = self.tokenizer.decode(token, skip_special_tokens=True)
        return word

    def encode_tokens(self):
        """
        Encode the input string to tokens.
        """
        raise NotImplementedError

    def load_model(self):
        """
        Load the model.
        """
        raise NotImplementedError

    def clear(self):
        """
        Clear the chat session.
        """
        raise NotImplementedError

    def update_history(self):
        """
        Update chat history.
        """
        raise NotImplementedError
