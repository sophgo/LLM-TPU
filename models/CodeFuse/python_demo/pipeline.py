import argparse

from BaseModel.base_model import BaseModel

class CodeFuse(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        self.prompt = (
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.EOS = self.tokenizer.im_end_id # tokenizer.encode("<|im_end|>")
        self.history = [self.system_prompt]

        # load model
        self.load_model(args)

    def load_model(self, args):
        if args.decode_mode == "jacobi":
            from CodeFuse.python_demo import chat_jacobi
            self.model = chat_jacobi.CodeFuse()
        elif args.decode_mode == "basic":
            from CodeFuse.python_demo import chat
            self.model = chat.CodeFuse()
            self.model.init(self.devices, args.model_path)
            self.model.temperature = args.temperature
            self.model.top_p = args.top_p
            self.model.repeat_penalty = args.repeat_penalty
            self.model.repeat_last_n = args.repeat_last_n
            self.model.max_new_tokens = args.max_new_tokens
            self.model.generation_mode = args.generation_mode
            self.model.prompt_mode = args.prompt_mode
        self.SEQLEN = self.model.SEQLEN

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

def main(args):
    model = CodeFuse(args)
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
    args = parser.parse_args()
    main(args)
