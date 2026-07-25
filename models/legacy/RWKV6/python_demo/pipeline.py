import argparse

import time
from rwkv_tokenizer import RWKV_TOKENIZER


class RWKV6:
    def __init__(self, args) -> None:
        # device_id
        self.devices = [int(d) for d in args.devid.split(",")]

        # load tokenizer
        print("Load " + args.tokenizer_path + " ...")
        self.tokenizer = RWKV_TOKENIZER(args.tokenizer_path)

        # warm up
        self.tokenizer.decode([[10]])

        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = """"""

        self.EOS = 0
        self.system = {"role": "system", "content": self.system_prompt}
        self.history = [self.system]
        self.enable_history = args.enable_history

        # load model
        self.load_model(args)

    def load_model(self, args):
        if args.decode_mode == "basic":
            import chat

            self.model = chat.RWKV6()
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

    def clear(self):
        self.history = [self.system]

    def test(self, input="Elon Musk has", test_num=50):
        tokens = self.tokenizer.encode(input)[0]
        input_len = len(tokens)
        # print(tokens)
        print(f"输入长度{len(tokens)} {input}")
        output_token = tokens
        ti = time.time()
        token = self.model.prefill(tokens, True, False, False)  # 缓存状态
        print(f"预填充耗时{time.time()-ti}")

        while token != 0 and len(output_token) < test_num + input_len:
            output_token.append(token)
            print(self.tokenizer.decode([[token]])[0], end="", flush=True)
            token = self.model.rnn_gen(False,False)
        if token != 0:
            print(self.tokenizer.decode([[token]])[0], end="\n\n", flush=True)

    # def update_history(self):
    #     if self.model.token_length >= self.SEQLEN:
    #         print("... (reach the maximal length)", flush=True, end="")
    #         self.history = [self.system]
    #     else:
    #         self.history.append({"role": "assistant", "content": self.answer_cur})


def main(args):
    model = RWKV6(args)
    input_str = """User: 
基于cpp编写一个冒泡排序算法
Assistant:
"""
    model.test(input_str)
    print()
    while True:
        input_str = input("输入文本:\n")
        if input_str != "exit" or input_str != "q":
            model.test(input_str)

    # model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_path", type=str, required=True, help="path to the bmodel file"
    )
    parser.add_argument(
        "-t",
        "--tokenizer_path",
        type=str,
        default="../support/token_config",
        help="path to the tokenizer file",
    )
    parser.add_argument("-d", "--devid", type=str, default="0", help="device ID to use")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature scaling factor for the likelihood distribution",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="cumulative probability of token words to consider as a set of candidates",
    )
    parser.add_argument(
        "--repeat_penalty", type=float, default=1.0, help="penalty for repeated tokens"
    )
    parser.add_argument(
        "--repeat_last_n",
        type=int,
        default=32,
        help="repeat penalty for recent n tokens",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="max new token length to generate",
    )
    parser.add_argument(
        "--generation_mode",
        type=str,
        choices=["greedy", "penalty_sample"],
        default="greedy",
        help="mode for generating next token",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        choices=["prompted", "unprompted"],
        default="prompted",
        help="use prompt format or original input",
    )
    parser.add_argument(
        "--decode_mode",
        type=str,
        default="basic",
        choices=["basic", "jacobi"],
        help="mode for decoding",
    )
    parser.add_argument(
        "--enable_history",
        action="store_true",
        help="if set, enables storing of history memory",
    )
    args = parser.parse_args()
    main(args)
