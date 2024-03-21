import time
from transformers import AutoTokenizer

class BaseModel:
    def __init__(self, model_path, tokenizer_path, devid, generation_mode="greedy", decode_mode="basic"):
        # preprocess parameters, such as prompt & tokenizer
        self.input_str = ""
        self.system_prompt = ""

        # model parameters
        self.token_length = 0
        self.SEQLEN = None
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        # devid
        if isinstance(devid, str):
            self.devices = [int(d) for d in devid.split(",")]
        elif isinstance(devid, int):
            raise ValueError("The input device should be in string format, like --devid '0' or --devid '0,1', but we received --devid {}".format(str(devid)))
        else:
            raise ValueError("The type of devis is wrong!")

        # postprocess parameters
        if generation_mode not in ["greedy","sample"]:
            raise ValueError("generation_mode should be in {}, but we get {}".format(["greedy","sample"], generation_mode))
        self.generation_mode = generation_mode
        if decode_mode not in ["basic","jacobi"]:
            raise ValueError("decode_mode should be in {}, but we get {}".format(["basic","jacobi"], decode_mode))
        self.decode_mode = decode_mode

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

                print("\nAnswer: ", end='')
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
        self.answer_cur = self.sp.decode(result_tokens, skip_special_tokens=True)
        print(self.answer_cur, end='')
        end = time.time()

        duration = end - start
        tps = len(result_tokens) / duration

        self.update_history()

        print()
        print(f"TPS: {tps:.3f} token/s")

    def stream_predict(self, query, messages=None):
        self.answer_cur = ""

        self.input_str = query
        tokens = self.generate_tokens()

        if not tokens:
            print("Sorry: your question is too wierd!!")
            return
        if self.token_length > self.SEQLEN:
            print("The maximum question length should be shorter than {} but we get {} instead.".format(self.SEQLEN, self.token_length))
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

