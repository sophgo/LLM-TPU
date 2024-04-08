import argparse

from BaseModel.base_model import BaseModel


class Qwen1_5(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = "You are a helpful assistant."
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.EOS = self.tokenizer.eos_token_id
        self.decode_mode = "diff"

        # load model
        self.load_model(args)

    def load_model(self, args):
        if len(self.devices) > 1:
            raise ValueError("not support now")
        else:
            from Qwen1_5.speculative_sample_demo import chat_speculative
            self.model = chat.Qwen()
            self.model.init(self.devices, args.draft_model_path, args.target_model_path)
            self.model.temperature = args.temperature
            self.model.top_p = args.top_p
            self.model.repeat_penalty = args.repeat_penalty
            self.model.repeat_last_n = args.repeat_last_n
            self.model.max_new_tokens = args.max_new_tokens
            self.model.generation_mode = args.generation_mode
            self.model.prompt_mode = args.prompt_mode
        self.SEQLEN = self.model.SEQLEN

    def clear(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def update_history(self):
        if self.token_length >= self.SEQLEN - 10:
            print("... (reach the maximal length)", flush=True, end="")
            self.messages = [self.messages[0]]
            self.messages.append({"role": "user", "content": self.input_str})
            self.messages.append({"role": "assistant", "content": self.answer_cur})
        else:
            self.messages.append({"role": "assistant", "content": self.answer_cur})

    def encode_tokens(self):
        self.messages.append({"role": "user", "content": self.input_str})
        text = self.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer(text).input_ids
        return tokens
    
    def chat(self):
        from time import time
        self.input_str = input("\nQuestion: ")
        tokens = self.encode_tokens()
        print(tokens)
        t1 = time()
        res = self.model.generate(tokens, 151645)
        t2 = time()
        print(t2 - t1)
        print(res)
        # self.model.deinit()
        breakpoint()

def main(args):
    model = Qwen1_5(args)
    model.chat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft_model_path', type=str, required=True, help='path to the draft bmodel file')
    parser.add_argument('--target_model_path', type=str, required=True, help='path to the target bmodel file')
    parser.add_argument('-t', '--tokenizer_path', type=str, default="../support/token_config", help='path to the tokenizer file')
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--enable_history', action='store_true', help="if set, enables storing of history memory.")
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')
    parser.add_argument('--top_p', type=float, default=1.0, help='cumulative probability of token words to consider as a set of candidates')
    parser.add_argument('--repeat_penalty', type=float, default=1.0, help='penalty for repeated tokens')
    parser.add_argument('--repeat_last_n', type=int, default=32, help='repeat penalty for recent n tokens')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new token length to generate')
    parser.add_argument('--generation_mode', type=str, choices=["penalty_sample"], default="penalty_sample", help='mode for generating next token')
    parser.add_argument('--prompt_mode', type=str, choices=["prompted", "unprompted"], default="prompted", help='use prompt format or original input')
    args = parser.parse_args()
    main(args)
