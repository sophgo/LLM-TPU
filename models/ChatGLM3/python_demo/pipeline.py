import argparse

from BaseModel.base_model import BaseModel


class ChatGLM(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        # preprocess parameters, such as prompt & tokenizer
        self.history = list()
        self.EOS = self.tokenizer.eos_token_id
        self.decode_mode = "diff"

        # load model
        self.load_model(args)

    def load_model(self, args):
        from ChatGLM3.python_demo import chat
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
        # self.history.append({"role": "user", "content": self.input_str})
        # text = self.tokenizer.apply_chat_template(
        #     self.history, tokenize=False, add_generation_prompt=True
        # )
        # tokens = self.tokenizer(text).input_ids
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
