import os
import time
import argparse
import pandas as pd
from transformers import AutoTokenizer

import chat_checked

import sys
sys.path.append("../../../harness/C-Eval")
from utils import load_json, dump_json, construct_prompt, extract_cot_answer


class Qwen2():
    def __init__(self, args):
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]

        # load tokenizer
        print("Load " + args.tokenizer_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )

        # warm up
        self.tokenizer.decode([0])

        # preprocess parameters, such as prompt & tokenizer
        self.EOS = self.tokenizer.eos_token_id
        self.enable_history = args.enable_history

        self.model = chat_checked.Qwen()
        self.init_params(args)
        self.model.init_decrypt()


    def load_model(self, model_path):
        load_start = time.time()
        self.model.init(self.devices, model_path)
        load_end = time.time()
        print(f"\nLoad Time: {(load_end - load_start):.3f} s")


    def init_params(self, args):
        self.model.temperature = args.temperature
        self.model.top_p = args.top_p
        self.model.repeat_penalty = args.repeat_penalty
        self.model.repeat_last_n = args.repeat_last_n
        self.model.max_new_tokens = args.max_new_tokens
        self.model.generation_mode = args.generation_mode
        self.model.lib_path = args.lib_path


    def encode_tokens(self, prompt):
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer(text).input_ids
        return tokens


    def stream_answer(self, tokens):
        """
        Stream the answer for the given tokens.
        """
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []

        # First token
        token = self.model.forward_first(tokens)

        # Following tokens
        while token != self.EOS and self.model.token_length < self.model.SEQLEN and self.model.status_code == 0 and tok_num < self.model.max_new_tokens:
            word = self.tokenizer.decode(token, skip_special_tokens=True)
            self.answer_token += [token]
            print(word, flush=True, end="")
            tok_num += 1
            token = self.model.forward_next()

        # handle exception
        if self.model.status_code < 0:
            return

        self.answer_cur = self.tokenizer.decode(self.answer_token)

        return self.answer_cur


    def test_ceval(self):
        """
        Test c-eval
        """
        self.system_prompt = "You will provide correct answer to the question."
        self.load_model(args.model_path)

        # handle exception
        if self.model.status_code < 0:
            return self.model.status_code

        test_path = "ceval-exam/test"
        subject_path = "subject_mapping.json"
        subject_map = load_json(subject_path)

        # 3. inference
        submit_path = "Qwen2_submit.csv"

        res = {}
        subject_num = len(os.listdir(test_path))
        print(f"Subject numbers: {subject_num}")
        for test_csv_file in os.listdir(test_path):
            test_csv_path = os.path.join(test_path, test_csv_file)
            test_df = pd.read_csv(test_csv_path)

            subject = test_csv_file.replace("_test.csv", "")
            subject_zh = subject_map[subject][1]

            subject_dict = {}
            print("======================================")
            print("======================================")
            print("Current subject:", subject)
            print("======================================")
            print("======================================")
            # if subject != "middle_school_physics":continue
            for i in range(len(test_df)):
                print(f"\n================={i}/{len(test_df)}====================")
                prompt = construct_prompt(subject_zh, [], test_df.loc[i], 0)
                tokens = self.encode_tokens(prompt)
                print("token length:", len(tokens))
                if len(tokens) >= 4096:
                    continue
                pred = self.stream_answer(tokens)
                
                if self.model.status_code < 0:
                    return self.model.status_code

                option = extract_cot_answer(pred)
                #print("\nprediction:", pred)
                print("\noption:", option)

                subject_dict[str(i)] = option
            res[subject] = subject_dict

        # 4. deinit & save
        dump_json(res, submit_path)

        # deinit
        self.model.deinit_decrypt()
        self.model.deinit()

        return self.model.status_code
    
    def test_sample(self):

        self.system_prompt = "You are a helpful assistant."
        self.load_model(args.model_path)

        # handle exception
        if self.model.status_code < 0:
            return self.model.status_code
        
        self.input_str = "hello"
        tokens = self.encode_tokens(self.input_str)
        # check tokens
        if not tokens:
            print("Sorry: your question is empty!!")
            return -1
        if len(tokens) > self.model.SEQLEN:
            print(
                "The maximum question length should be shorter than {} but we get {} instead.".format(
                    self.model.SEQLEN, len(tokens)
                )
            )
            return -1

        print("\nAnswer: ", end="")
        self.stream_answer(tokens)

        return 0

"""
-1: your input is empty or exceed the maximum length
-2: can not to create handle
-3: can not to create bmrt
-4: can not to load bmodel, maybe your key is wrong
-5: can not to inference bmodel
"""
def main(args):
    # # test c-eval
    # model = Qwen2(args)
    # status_code = model.test_ceval()
    

    # test chat
    model = Qwen2(args)
    status_code = model.test_sample()

    print("Status Code: ", status_code)


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
    parser.add_argument('--enable_history', action='store_true', help="if set, enables storing of history memory")
    parser.add_argument('--lib_path', type=str, default='', help='lib path by user')
    args = parser.parse_args()
    main(args)
