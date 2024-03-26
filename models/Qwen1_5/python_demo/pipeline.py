import argparse

from BaseModel.base_model import BaseModel


class Qwen1_5(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = "You are a helpful assistant."
        self.SEQLEN = 512
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.decode_mode = "diff"

        # load model
        self.load_model(args)

    def load_model(self, args):
        if len(self.devices) > 1:
            from Qwen1_5.python_demo import chat_parallel
            self.model = chat_parallel.Qwen()
            self.model.init(
                self.devices,
                self.sp.im_end_id,
                args.model_path
            )
        elif args.generation_mode in ["greedy", "sample"]:
            from Qwen1_5.python_demo import chat
            self.model = chat.Qwen()
            self.model.init(
                self.devices,
                args.model_path,
                args.temperature,
                args.top_p,
                args.max_new_tokens,
                args.generation_mode,
                args.prompt_mode,
            )

    def clear(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def update_history(self):
        if self.token_length >= self.SEQLEN - 128:
            print("... (reach the maximal length)", flush=True, end="")
            self.messages = [self.messages[0]]
            self.messages.append({"role": "user", "content": self.input_str})
            self.messages.append({"role": "assistant", "content": self.answer_cur})
        else:
            self.messages.append({"role": "assistant", "content": self.answer_cur})

    def encode_tokens(self):
        self.messages.append({"role": "user", "content": self.input_str})
        text = self.sp.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        tokens = self.sp(text).input_ids
        return tokens


def main(args):
    model = Qwen1_5(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='path to the bmodel file')
    parser.add_argument('--tokenizer_path', type=str, default="../support/token_config", help='path to the tokenizer file')
    parser.add_argument('--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')
    parser.add_argument('--top_p', type=float, default=1.0, help='cumulative probability of token words to consider as a set of candidates')
    parser.add_argument('--repeat_penalty', type=float, default=1.0, help='penalty for repeated tokens')
    parser.add_argument('--repeat_last_n', type=int, default=32, help='repeat penalty for recent n tokens')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new token length to generate')
    parser.add_argument('--generation_mode', type=str, choices=["greedy", "sample"], default="sample", help='mode for generating next token')
    parser.add_argument('--prompt_mode', type=str, choices=["prompted", "unprompted"], default="prompted", help='use prompt format or original input')

    args = parser.parse_args()
    main(args)
