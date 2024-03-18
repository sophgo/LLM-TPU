import argparse

import chat
from BaseModel.base_model import BaseModel

class Qwen1_5(BaseModel):
    def __init__(self, model_path, tokenizer_path, devid, generation_mode="greedy"):
        super().__init__(model_path, tokenizer_path, devid, generation_mode)
        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = "You are a helpful assistant."
        self.SEQLEN = 512
        self.messages = [{"role": "system", "content": self.system_prompt}]

        # load model
        self.load_model()

    def load_model(self):
        self.model = chat.Qwen()
        self.model.init(self.devices, self.sp.eos_token_id, self.model_path)

    def clear(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def update_history(self):
        if self.token_length >= self.SEQLEN - 128:
            print("... (reach the maximal length)", flush=True, end='')
            self.messages = [self.messages[0]]
            self.messages.append({"role": "user", "content": self.input_str})
            self.messages.append({"role": "assistant", "content": self.answer_cur})
        else:
            self.messages.append({"role": "assistant", "content": self.answer_cur})

    def generate_tokens(self):
        self.messages.append({"role":"user","content":self.input_str})
        text = self.sp.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        tokens = self.sp(text).input_ids
        return tokens

def main(args):
    model = Qwen1_5(args.model_path, args.tokenizer_path, args.devid, args.generation_mode)
    model.chat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--devid', type=str, default='0', help='device ID to use.')
    parser.add_argument('--model_path', type=str, required=True, help='path to the bmodel file.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='path to the tokenizer file.')
    parser.add_argument('--generation_mode', type=str, default="greedy", help='mode for generating next token.')
    args = parser.parse_args()
    main(args)

