import os
import time
import argparse
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer

from utils import load_json, dump_json, record_time, construct_prompt, load_model, inference_model


def main(args):
    # 1. define params
    example_num = 0
    dev_path = "ceval-exam/dev"
    test_path = "ceval-exam/test"
    subject_path = "subject_mapping.json"
    subject_map = load_json(subject_path)

    # 2. create engine
    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    # init model
    model = load_model(args)

    # 3. inference
    submit_path = f"{args.model_name}_{args.device}_{args.max_new_tokens}_submit.csv"

    res = {}
    subject_num = len(os.listdir(test_path))
    print(f"Subject numbers: {subject_num}")
    count = 0
    cost_time = {}
    for dev_csv_file, test_csv_file in zip(os.listdir(dev_path), os.listdir(test_path)):
        t_start = time.time()
        count = count + 1
        dev_csv_path = os.path.join(dev_path, dev_csv_file)
        test_csv_path = os.path.join(test_path, test_csv_file)
        dev_df = pd.read_csv(dev_csv_path)
        test_df = pd.read_csv(test_csv_path)

        subject = test_csv_file.replace("_test.csv", "")
        subject_zh = subject_map[subject][1]
        dev_row = [dev_df.loc[i] for i in range(example_num)]

        subject_dict = {}
        print("======================================")
        print("======================================")
        print("Current subject:", subject)
        print("subject no: ", count)
        print("======================================")
        print("======================================")
        # if subject != "middle_school_physics":continue
        for i in tqdm(range(len(test_df))):
            prompt = construct_prompt(subject_zh, dev_row, test_df.loc[i], example_num)
            pred = inference_model(model, tokenizer, prompt, args)
            print("prediction:", pred)
            subject_dict[str(i)] = pred
        if args.device == "cuda":time.sleep(10)
        res[subject] = subject_dict
        cost_time[subject] = time.time() - t_start

    # 4. deinit & save
    dump_json(res, submit_path)
    record_time(cost_time, f"{args.model_name}_time.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--devid', required=True, type=str, default='0', help='device ID to use')
    parser.add_argument('--model_path', required=True, type=str, help='path to the bmodel file')
    parser.add_argument('--tokenizer_path', required=True, type=str, help='path to the tokenizer file')
    parser.add_argument("--device", required=True, type=str, choices=["cuda", "tpu"], default="tpu")
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')
    parser.add_argument('--top_p', type=float, default=1.0, help='cumulative probability of token words to consider as a set of candidates')
    parser.add_argument('--repeat_penalty', type=float, default=1.0, help='penalty for repeated tokens')
    parser.add_argument('--repeat_last_n', type=int, default=32, help='repeat penalty for recent n tokens')
    parser.add_argument('--max_new_tokens', type=int, default=1, help='max new token length to generate')
    parser.add_argument('--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
    parser.add_argument('--model_name', required=True, type=str, choices=["ChatGLM3", "Qwen1_5", "Qwen2"], help='the model to evaluate')

    args = parser.parse_args()
    main(args)
