import os
import json
import time
import random
import argparse
from transformers import AutoTokenizer

import sys
import chat
sys.path.append("../../../harness/C-Eval")
from utils import load_json, dump_json, construct_prompt, extract_cot_answer


class Qwen:
    def __init__(self, args):
        # preprocess parameters, such as prompt & tokenizer
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]
        self.model_path = args.model_path

        # load tokenizer
        print("Load " + args.tokenizer_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )

        # warm up
        self.tokenizer.decode([0])
        self.EOS = self.tokenizer.eos_token_id

        self.model = chat.Qwen()
        self.init_params(args)


        self.seq_length_list = [10240,8192,7168,6144,5120,4096,3072,2048,1024]
        self.share_length_list = [8192,7680,7168,6144,5120,4096,3072,2048,1024]

    def load_model(self, model_path, read_bmodel):
        load_start = time.time()
        self.model.init(self.devices, model_path, read_bmodel) # when read_bmodel = false, not to load weight, reuse weight
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
        self.model.embedding_path = args.embedding_path

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

    def stream_answer(self, tokens, inference_mode, max_tok_num):
        """
        Stream the answer for the given tokens.
        """
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []

        print()
        # First token
        first_start = time.time()
        if inference_mode == "normal":
            token = self.model.forward_first(tokens)
        elif inference_mode == "share":
            token = self.model.forward_unshare(tokens)
        else:
            raise ValueError(f"Not support {inference_mode}")
        first_end = time.time()
        # Following tokens
        while (max_tok_num > 0 and tok_num < max_tok_num) or (
            max_tok_num == 0
            and token != self.EOS
            and self.model.total_length < self.model.SEQLEN
        ):
            word = self.tokenizer.decode(token, skip_special_tokens=True)
            self.answer_token += [token]
            print(word, flush=True, end="")
            tok_num += 1
            token = self.model.forward_next()
        self.answer_cur = self.tokenizer.decode(self.answer_token)

        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration

        print()
        if inference_mode == "normal":
            print(f"FTL Time: {first_duration:.3f} s")
        elif inference_mode == "share":
            print(f"Unshare FTL Time: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

    def read_json(self, json_path, task_id):
        with open(json_path, "r") as file:
            text = json.load(file)
        system_str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"

        content_str = ""
        if "content" in text[task_id]:
            content_str = system_str + text[task_id]["content"]
        question_str = text[task_id]["question"] + "<|im_end|>\n<|im_start|>assistant\n"
        return content_str, question_str

    def get_seq_index(self, total_length, in_length):
        seq_index = []
        for index, (t_length, i_length) in enumerate(zip(self.seq_length_list, self.share_length_list)):
            if t_length >= total_length and i_length >= in_length:
                seq_index.append(index)
        return seq_index

    def test_length(self):
        json_path = "../../../assets/long_case.json"
        input_str = load_json(json_path)[0]["content"]

        tokens = self.tokenizer.encode(input_str)

        self.model.init_decrypt()
        self.load_model(args.model_path, read_bmodel=True)

        for i in range(120, self.model.SEQLEN - 10):
            self.model.stage_idx = i % 2
            self.load_model(args.model_path, read_bmodel=False)
            print(f"\n----------------------Length : {i}----------------------")
            self.stream_answer(tokens[:i], "normal", 5)

        # deinit
        self.model.deinit_decrypt()
        self.model.deinit()
        return

    def test_sample(self):
        json_path = "../../../assets/sophgo_kv_cache_share_test_case.json"
        share_str, unshare_str_0 = self.read_json(json_path, 0)
        _, unshare_str_1 = self.read_json(json_path, 1)
        _, unshare_str_2 = self.read_json(json_path, 2)
        # share_str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        # unshare_str_0 = "can you help me<|im_end|>\n<|im_start|>assistant\n"
        # unshare_str_1 = "tell me a love story<|im_end|>\n<|im_start|>assistant\n"
        # unshare_str_2 = "tell me a love story<|im_end|>\n<|im_start|>assistant\n"

        self.model.init_decrypt()
        # ===------------------------------------------------------------===
        # Model 0
        # ===------------------------------------------------------------===
        # load model 0
        self.model.prefill_reuse = 0
        self.model.stage_idx = 0
        self.load_model(self.model_path, read_bmodel=True)

        # share prefill
        share_tokens = self.tokenizer.encode(
            share_str, max_length=8000, truncation=True
        )

        # task 0
        # first + decode
        unshare_tokens = self.tokenizer.encode(unshare_str_0)
        self.stream_answer(share_tokens + unshare_tokens, "normal", 0)


        # task 1
        # first + decode
        unshare_tokens = self.tokenizer.encode(unshare_str_0)
        self.stream_answer(share_tokens + unshare_tokens, "normal", 0)

        # ===------------------------------------------------------------===
        # Model 1
        # ===------------------------------------------------------------===
        # load model 1
        self.model.prefill_reuse = 0
        self.model.stage_idx = 1
        self.load_model(self.model_path, read_bmodel=False)

        # first + decode
        for i in range(9):
            unshare_tokens = self.tokenizer.encode(unshare_str_0)
            self.stream_answer(share_tokens[:4000] + unshare_tokens, "normal", 0)

        # ===------------------------------------------------------------===
        # Deinit
        # ===------------------------------------------------------------===
        self.model.deinit_decrypt()
        self.model.deinit()

    def test_random(self):
        json_path = "../../../assets/long_case.json"
        share_str, unshare_str_0 = self.read_json(json_path, 0)
        _, unshare_str_1 = self.read_json(json_path, 1)
        _, unshare_str_2 = self.read_json(json_path, 2)

        self.model.init_decrypt()
        # ===------------------------------------------------------------===
        # Model 0
        # ===------------------------------------------------------------===
        # load model 0
        self.model.prefill_reuse = 0
        self.model.stage_idx = 0
        self.load_model(self.model_path, read_bmodel=True)

        # share prefill
        for i in range(10):
            in_length = random.randint(500, 8192)
            out_length = random.randint(200, 512)
            in_tokens = self.tokenizer.encode(
                share_str, max_length=in_length, truncation=True
            )
            unshare_tokens = self.tokenizer.encode(unshare_str_0)

            in_length = in_length + len(unshare_tokens)
            total_length = in_length + out_length

            seq_index = self.get_seq_index(total_length, in_length)
            self.model.stage_idx = seq_index[-1]
            self.load_model(self.model_path, read_bmodel=False)
            self.stream_answer(in_tokens[:in_length - len(unshare_tokens)] + unshare_tokens, "normal", out_length)

        # ===------------------------------------------------------------===
        # Deinit
        # ===------------------------------------------------------------===
        self.model.deinit_decrypt()
        self.model.deinit()

    def test_ceval(self):
        """
        Test c-eval
        """
        import pandas as pd
        self.system_prompt = "You will provide correct answer to the question."

        test_path = "ceval-exam/test"
        subject_path = "subject_mapping.json"
        subject_map = load_json(subject_path)

        # 3. inference
        self.model.init_decrypt()
        submit_path = "Qwen2_submit.json"
        self.model.stage_idx = 0
        self.load_model(self.model_path, read_bmodel=True)

        res = {}
        subject_num = len(os.listdir(test_path))
        print(f"Subject numbers: {subject_num}")
        for idx, test_csv_file in enumerate(os.listdir(test_path)):
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
                in_length = len(tokens)
                print("token length:", in_length)
                if in_length >= 3200:
                    raise ValueError(f"The length you input is {in_length}, exceed the maximum length")

                seq_index = self.get_seq_index(in_length + self.model.max_new_tokens, in_length)
                self.model.stage_idx = seq_index[-1]
                self.load_model(self.model_path, read_bmodel=False)
                self.stream_answer(tokens, "normal", self.model.max_new_tokens)

                option = extract_cot_answer(self.answer_cur)
                #print("\nprediction:", pred)
                print("\noption:", option)

                subject_dict[str(i)] = option
            res[subject] = subject_dict

        # 4. deinit & save
        dump_json(res, submit_path)

        # deinit
        self.model.deinit_decrypt()
        self.model.deinit()


"""
-1: your input is empty or exceed the maximum length
-2: can not to create handle
-3: can not to create bmrt
-4: can not to load bmodel, maybe your key is wrong
-5: can not to inference bmodel
-6: addr_mode = 0, but must set addr_mode =1
"""
def main(args):
    start_time = time.time()

    try:
        engine = Qwen(args)

        # 1. test one sample
        # engine.test_sample()

        # 2. test random
        # engine.test_random()
        
        # 2. test c-eval
        engine.test_ceval()

        # 3. test length
        # engine.test_length()


        print("All Right!")
    except RuntimeError:
        print("RuntimeError")
    except ValueError:
        print("ValueError")
    except:
        print("Error")

    end_time = time.time()
    print(f"\nTotal Time: {(end_time - start_time):.3f} s")
    print("Status Code: ", engine.model.status_code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True, help="path to the bmodel")
    parser.add_argument('-t', '--tokenizer_path', type=str, default="../support/token_config", help='path to the tokenizer file')
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')
    parser.add_argument('--top_p', type=float, default=1.0, help='cumulative probability of token words to consider as a set of candidates')
    parser.add_argument('--repeat_penalty', type=float, default=1.2, help='penalty for repeated tokens')
    parser.add_argument('--repeat_last_n', type=int, default=32, help='repeat penalty for recent n tokens')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new token length to generate')
    parser.add_argument('--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
    parser.add_argument('--prompt_mode', type=str, choices=["prompted", "unprompted"], default="prompted", help='use prompt format or original input')
    parser.add_argument('--enable_history', action='store_true', help="if set, enables storing of history memory")
    parser.add_argument('--lib_path', type=str, default='', help='lib path by user')
    parser.add_argument('--embedding_path', type=str, default='', help='binary embedding path')
    args = parser.parse_args()
    main(args)
