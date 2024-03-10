import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from ChatGLM3.python_demo import chat

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def main(args):
    # 1. define params
    example_num = 0
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    # 2. create engine
    devices = [int(d) for d in args.devid.split(",")]
    engine = chat.ChatGLM()
    engine.init(devices, args.model_path, args.tokenizer_path)


    # 3. construct prompt & inference
    all_cors = []
    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: example_num]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors = []
        for i in tqdm(range(len(test_df))):
            prompt_end = format_example(test_df, i, include_answer=False)
            few_shot_prompt = gen_prompt(dev_df, subject, example_num)
            prompt = few_shot_prompt + prompt_end
            pred = engine.predict_option(prompt)
            label = test_df.iloc[i, test_df.shape[1] - 1]
            cors.append(pred == label)
        weighted_acc = np.mean(cors)
        print("Average accuracy: {:.3f}".format(weighted_acc))
        all_cors.append(cors)

    # deinit & compute acc
    engine.deinit()
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument('--devid', type=str, help='Device ID to use.')
    parser.add_argument('--model_path', type=str, help='Path to the bmodel file.')
    parser.add_argument('--tokenizer_path', type=str, help='Path to the tokenizer file.')
    args = parser.parse_args()
    main(args)
