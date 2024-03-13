import argparse
from models import BaseModel

class Qwen1_5(BaseModel):
    def __init__(self, bmodel_path, tokenizer_path, devid, model="greedy"):
        super().__init__(bmodel_path, tokenizer_path, devid, model)
        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = "You are a helpful assistant."
        self.SEQLEN = 512

    def generate_tokens(self):
        text = self.sp.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        tokens = self.sp(text).input_ids
        return tokens

def main(args):
    model = Qwen1_5(args.model_path, args.tokenizer_path, args.devid, args.sample_mode)
    model.chat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--devid', type=str, default=0, help='Device ID to use.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the bmodel file.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer file.')
    parser.add_argument('--sample_mode', type=str, default="greedy", help='mode for sampling next token.')
    args = parser.parse_args()
    main(args)

