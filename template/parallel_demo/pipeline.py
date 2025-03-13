import os
import sys
import json
import time
import argparse
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

import chat
from demo.pipeline import Model


class MultiDeviceModel(Model):
    def __init__(self, args):
        # test
        self.test_input = None
        self.test_media = None

        # preprocess parameters, such as prompt & tokenizer
        self.devices = [int(d) for d in args.devid.split(",")]
        config_path = os.path.join(args.dir_path, "config.json")
        self.tokenizer_path = os.path.join(args.dir_path, "tokenizer")

        # config
        with open(config_path, 'r') as file:
            self.config = json.load(file)

        # Initialize model
        self.model_type = args.model_type if args.model_type else self.config['model_type']
        self.model = chat.Model()
        self.init_params(args)

        # Initialize model-specific mapper dynamically
        self.map(args)

        # warm up
        self.tokenizer.decode([0])
        self.init_history()

        # load model
        self.load_model(args, read_bmodel=True)

    def load_model(self, args, read_bmodel):
        bmodel_files = [f for f in os.listdir(args.dir_path) if f.endswith('.bmodel')]
        if len(bmodel_files) > 1:
            raise RuntimeError(f"Found multiple bmodel files in {args.dir_path}, please specify one with --model_path")
        elif not bmodel_files:
            raise FileNotFoundError(f"No bmodel files found in {args.dir_path}")
        model_path = os.path.join(args.dir_path, bmodel_files[0])

        load_start = time.time()
        self.model.init(self.devices, model_path)
        load_end = time.time()
        print(f"\nLoad Time: {(load_end - load_start):.3f} s")

    def init_params(self, args):
        self.model.NUM_LAYERS = self.config["num_hidden_layers"]
        self.enable_history = args.enable_history

    def prefill_phase(self, text, media_path, media_type):
        print("\n回答: ", end="")
        first_start = time.time()

        tokens = self.encode_tokens(text, media_path, media_type)

        token = self.model.forward_first(tokens)

        first_end = time.time()
        self.ftl = first_end - first_start
        return token

    def decode_phase(self, token):
        """
        Stream the answer for the given tokens.
        """
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []
        next_start = time.time()

        # Following tokens
        full_word_tokens = []
        while token not in self.EOS and self.model.token_length < self.model.SEQLEN:
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "�" in word:
                token = self.model.forward_next(token)
                tok_num += 1
                continue
            self.answer_token += full_word_tokens
            print(word, flush=True, end="")
            tok_num += 1
            full_word_tokens = []
            token = self.model.forward_next(token)

        # counting time
        next_end = time.time()
        next_duration = next_end - next_start
        self.tps = tok_num / next_duration

        if self.enable_history:
            self.answer_cur = self.tokenizer.decode(self.answer_token)
            self.update_history()
        else:
            self.init_history()

        print()
        print(f"FTL: {self.ftl:.3f} s")
        print(f"TPS: {self.tps:.3f} token/s")

def main(args):
    model = MultiDeviceModel(args)
    model.chat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--dir_path", type=str, default="./tmp",
                        help="dir path to the bmodel/config/tokenizer")
    parser.add_argument('-d', '--devid', type=str,
                        help='device ID to use')
    parser.add_argument('--enable_history', action='store_true',
                        help="if set, enables storing of history memory")
    parser.add_argument('--model_type', type=str, help="model type")
    args = parser.parse_args()
    main(args)
