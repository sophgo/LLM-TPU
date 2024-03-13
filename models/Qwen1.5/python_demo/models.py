import time
from transformers import AutoTokenizer
import chat

class BaseModel:
    def __init__(self, bmodel_path, tokenizer_path, devid, mode="greedy"):
        # preprocess parameters, such as prompt & tokenizer
        self.input_str = ""
        self.system_prompt = ""
        self.messages = [{"role": "system", "content": self.system_prompt}]

        # model parameters
        self.token_length = 0
        self.SEQLEN = None
        self.bmodel_path = bmodel_path
        self.tokenizer_path = tokenizer_path
        self.devid = devid

        # postprocess parameters
        if mode not in ["greedy","sample"]:
            raise ValueError("mode should be in {}, but we get {}".format(["greedy","sample"],mode))
        self.mode = mode

        # load tokenizer
        print("Load " + self.tokenizer_path + " ...")
        self.sp = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        self.EOS = self.sp.eos_token_id

        # load model
        devices = [int(d) for d in self.devid.split(",")]
        self.model = chat.Qwen()
        self.model.init(devices, self.sp.eos_token_id, self.bmodel_path)

        # warm up
        self.sp.decode([0]) 
        print("Done!")

    def generate_tokens(self):
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
                self.messages = [{"role": "system", "content": self.system_prompt}]
                continue
            # Chat
            else:
                # tokens_with_template = self.generate_tokens(self.input_str)
                self.messages.append({"role":"user","content":self.input_str})
                tokens = self.generate_tokens()

                print("\nAnswer: ")
                self.stream_answer(tokens)

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
        while token != self.EOS and self.token_length < self.SEQLEN:
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

        if self.token_length >= self.SEQLEN - 128:
            print("... (reach the maximal length)", flush=True, end='')
            self.messages = self.messages[0]
            self.messages.append({"role": "user", "content": self.input_str})
            self.messages.append({"role": "assistant", "content": self.answer_cur})
        else:
            self.messages.append({"role": "assistant", "content": self.answer_cur})

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

    def forward_first(self, tokens):
        if self.mode == "greedy":
            token = self.model.forward_first(tokens)
        elif self.mode == "sample":
            token = self.model.forward_first_with_topk(tokens, self.mode)
        return token
    
    def forward_next(self, token):
        if self.mode == "greedy":
            token = self.model.forward_next(token)
        elif self.mode == "sample":
            token = self.model.forward_next_with_topk(token, self.mode)
        return token

