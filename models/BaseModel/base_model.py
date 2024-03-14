import time
from transformers import AutoTokenizer

class BaseModel:
    def __init__(self, model_path, tokenizer_path, devid, generation_mode="greedy"):
        # preprocess parameters, such as prompt & tokenizer
        self.input_str = ""
        self.system_prompt = ""

        # model parameters
        self.token_length = 0
        self.SEQLEN = None
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.devid = devid

        # postprocess parameters
        if generation_mode not in ["greedy","sample"]:
            raise ValueError("generation_mode should be in {}, but we get {}".format(["greedy","sample"], generation_mode))
        self.generation_mode = generation_mode

        # load tokenizer
        print("Load " + self.tokenizer_path + " ...")
        self.sp = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)

        # warm up
        self.sp.decode([0]) 
        print("Done!")

    def generate_tokens(self):
        pass

    def load_model(self):
        pass

    def clear(self):
        pass

    def history_update(self):
        pass

    def chat(self):
        # Instruct:
        print(
f"""\n===========================================================
1. If you want to quit, please entry one of [q, quit, exit]
2. To create new chat-session, please entry one of [clear, new]
===========================================================""")
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
                tokens = self.generate_tokens()

                print("\nAnswer: ")
                self.stream_answer(tokens)

    def answer(self, tokens):
        self.answer_cur = ""

        if not tokens:
            print("Sorry: your question is too wierd!!")
            return
        if self.token_length > self.SEQLEN:
            print("The maximum question length should be shorter than {} but we get {} instead.".format(self.SEQLEN, self.token_length))
            return
        
        # Inference
        start = time.time()
        result_tokens = self.model.answer(tokens)
        self.answer_cur = self.sp.decode(result_tokens)
        print(self.answer_cur, end='')
        end = time.time()

        duration = end - start
        tps = len(result_tokens) / duration

        self.update_history()

        print()
        print(f"TPS: {tps:.3f} token/s")

    def stream_answer(self, tokens):
        tok_num = 0
        self.answer_cur = ""

        if not tokens:
            print("Sorry: your question is too wierd!!")
            return
        if self.token_length > self.SEQLEN:
            print("The maximum question length should be shorter than {} but we get {} instead.".format(self.SEQLEN, self.token_length))
            return
        
        # First token
        first_start = time.time()
        token = self.forward_first(tokens)
        first_end = time.time()

        # Following tokens
        while token != self.sp.eos_token_id and self.token_length < self.SEQLEN:
            diff = self.sp.decode([token])
            self.answer_cur += diff
            print(diff, flush=True, end='')
            if self.token_length < self.SEQLEN:
                self.token_length += 1
            tok_num += 1
            token = self.forward_next(token)
        
        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration

        self.update_history()

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

    def forward_first(self, tokens):
        if self.generation_mode == "greedy":
            token = self.model.forward_first(tokens)
        elif self.generation_mode == "sample":
            token = self.model.forward_first_with_topk(tokens, self.generation_mode)
        return token
    
    def forward_next(self, token):
        if self.generation_mode == "greedy":
            token = self.model.forward_next(token)
        elif self.generation_mode == "sample":
            token = self.model.forward_next_with_topk(token, self.generation_mode)
        return token

