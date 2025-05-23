import argparse
import time
from transformers import AutoTokenizer


class Megrez():
    def __init__(self, args):
        self.devices = args.devid
        # load tokenizer
        print("Load " + args.tokenizer_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        # warm up
        self.tokenizer.decode([0])

        # preprocess parameters, such as prompt & tokenizer
        self.ID_EOS = [120005, 120000, 120025]
        self.system_prompt = "You are a helpful assistant."
        self.system = {"role":"system", "content":self.system_prompt}
        self.history = [self.system]

        # load model
        self.load_model(args)

    def load_model(self, args):
        import chat
        self.model = chat.Megrez()
        self.model.init(self.devices, args.model_path)
        self.SEQLEN = self.model.SEQLEN

    def clear(self):
        self.history = [self.system]

    def update_history(self):
        if self.model.token_length >= self.SEQLEN:
            print("... (reach the maximal length)", flush=True, end='')
            self.clear
        else:
            self.history.append({"role":"assistant","content":self.answer_cur})

    def encode_tokens(self):
        self.clear
        self.history.append({"role": "user", "content": self.input_str})
        tokens = self.tokenizer.apply_chat_template(self.history, tokenize=True, add_generation_prompt=True)
        return tokens

    def chat(self):
        print(
            """\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
================================================================="""
        )
        # Stop Chatting with "exit" input
        while True:
            self.input_str = input("\nQuestion: ")

            if self.input_str in ["exit", "q", "quit"]:
                break
            elif self.input_str in ["clear", "new"]:
                self.clear()
            else:
                tokens = self.encode_tokens()
            # Chat
            first_start = time.time()
            token = self.model.forward_first(tokens)
            first_end = time.time()
            tok_num = 0
            # Following tokens
            full_word_tokens = []
            while token not in self.ID_EOS and self.model.token_length < self.SEQLEN:
                full_word_tokens.append(token)
                word = self.tokenizer.decode(
                    full_word_tokens, skip_special_tokens=True)
                if "�" not in word:
                    if len(full_word_tokens) == 1:
                        pre_word = word
                        word = self.tokenizer.decode([token, token], skip_special_tokens=True)[
                            len(pre_word):]
                    print(word, flush=True, end="")
                    full_word_tokens = []
                tok_num += 1
                token = self.model.forward_next()
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")

def main(args):
    model = Megrez(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
    parser.add_argument('-t', '--tokenizer_path', type=str, default="./support/token_config", help='path to the tokenizer file')
    parser.add_argument('-d', '--devid', type=int, default=0, help='device ID to use')
    args = parser.parse_args()
    main(args)