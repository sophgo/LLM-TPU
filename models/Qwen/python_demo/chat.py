import argparse

from BaseModel.base_model import BaseModel

class Qwen(BaseModel):
    def __init__(self, model_path, tokenizer_path, devid, generation_mode="greedy", decode_mode="jacobi"):
        super().__init__(model_path, tokenizer_path, devid, generation_mode, decode_mode)
        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        self.prompt = (
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.SEQLEN = 8192
        self.sp.eos_token_id = 151645 # tokenizer.encode("<|im_end|>")
        self.messages = [self.system_prompt]

        # load model
        self.load_model()

    def load_model(self):
        if self.decode_mode == "jacobi":
            import chat_jacobi
            self.model = chat_jacobi.Qwen()
        self.model.init(self.devices, self.sp.eos_token_id, self.model_path)

    def clear(self):
        self.messages = [self.system_prompt]

    def update_history(self):
        if self.token_length >= self.SEQLEN - 128:
            print("... (reach the maximal length)", flush=True, end='')
            self.messages = [self.messages[0]]
            self.messages.append(self.prompt.format(self.input_str) + self.answer_cur)
        else:
            self.messages[-1] = self.messages[-1] + self.answer_cur

    def generate_tokens(self):
        self.messages.append(self.prompt.format(self.input_str))
        text = "".join(self.messages)
        tokens = self.sp(text).input_ids
        return tokens

def main(args):
    model = Qwen(args.model_path, args.tokenizer_path, args.devid, args.generation_mode, args.decode_mode)
    model.chat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--devid', type=str, default=0, help='device ID to use.')
    parser.add_argument('--model_path', type=str, required=True, help='path to the bmodel file.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='path to the tokenizer file.')
    parser.add_argument('--generation_mode', type=str, default="greedy", choices=["greedy", "sample"], help='mode for generating next token.')
    parser.add_argument('--decode_mode', type=str, default="basic", choices=["basic", "jacobi"], help='mode for decoding.')
    args = parser.parse_args()
    main(args)
