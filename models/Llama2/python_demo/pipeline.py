import argparse

from BaseModel.base_model import BaseModel

class Llama2(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = '''You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.'''
        self.EOS = self.tokenizer.eos_token_id
        self.system = {"role":"system","content":self.system_prompt}
        self.history = [self.system]
        self.output_style = " "

        # load model
        self.load_model(args)

    def load_model(self, args):
        if args.decode_mode == "basic":
            import chat
            self.model = chat.Llama2()
            self.model.init(self.devices, args.model_path)
            self.model.temperature = args.temperature
            self.model.top_p = args.top_p
            self.model.repeat_penalty = args.repeat_penalty
            self.model.repeat_last_n = args.repeat_last_n
            self.model.max_new_tokens = args.max_new_tokens
            self.model.generation_mode = args.generation_mode
            self.model.prompt_mode = args.prompt_mode
        else:
            raise ValueError("decode mode: {} is illegal!".format(args.decode_mode))
        self.SEQLEN = self.model.SEQLEN

    def update_history(self):
        if self.model.token_length >= self.SEQLEN:
            print("... (reach the maximal length)", flush=True, end='')
            self.history = [self.system]
        else:
            self.history.append({"role":"assistant","content":self.answer_cur})

    def encode_tokens(self):
        tokens = []
        for item in self.history:
            content = item["content"]
            if item["role"] == "system":
                tokens.extend(self.tokenizer.encode("<s>[INST] <<SYS>>\n{}\n <</SYS>>".format(content), add_special_tokens = False))
            elif item["role"] == "user":
                tokens.extend(self.tokenizer.encode(content + " [/INST]", add_special_tokens = False))
            elif item["role"] == "assistant":
                tokens.extend(self.tokenizer.encode(content + " </s><s>", add_special_tokens = False))
            else:
                raise ValueError("role should be in {system , user assitent} but we get {}".format(item["role"]))
        tokens.extend(self.tokenizer.encode("[INST]" + self.input_str + " [/INST] ", add_special_tokens = False))
        self.history.append({"role":"user","content":self.input_str})
        return tokens

def main(args):
    model = Llama2(args)
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
