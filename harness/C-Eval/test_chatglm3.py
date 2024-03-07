import os
import json
import argparse
from tqdm import tqdm
import pandas as pd

from ChatGLM3.python_demo import chat

def load_json(json_path):
    with open(json_path, 'r') as f:
        res = json.load(f)
    return res

def dump_json(dic, json_path):
    with open(json_path, 'w') as json_file:
        json.dump(dic, json_file)
    return

def construct_content(subject, dev_row, test_row, example_num):
    sys_pattern = "以下是中国关于{}考试的单项选择题，请选出其中的正确答案。\n\n"
    question_pattern = "{}\nA. {}\nB. {}\nC. {}\nD. {}\n答案：{}\n"
    test_pattern = "{}\nA. {}\nB. {}\nC. {}\nD. {}\n答案："

    res = sys_pattern.format(subject)
    for i in range(example_num):
        res = res + question_pattern.format(dev_row[i].question, dev_row[i].A, dev_row[i].B, dev_row[i].C, dev_row[i].D, dev_row[i].anwser)
    res = res + test_pattern.format(test_row.question, test_row.A, test_row.B, test_row.C, test_row.D)
    return res

def main():
    # 1. define params
    example_num = 0
    dev_path = "ceval-exam/dev"
    test_path = "ceval-exam/test"
    submit_path = "submisstion.json"
    subject_path = "subject_mapping.json"
    subject_map = load_json(subject_path)

    # 2. read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--devid', type=str, help='Device ID to use.')
    parser.add_argument('--model_path', type=str, help='Path to the bmodel file.')
    parser.add_argument('--tokenizer_path', type=str, help='Path to the tokenizer file.')
    args = parser.parse_args()

    # 3. create engine
    devices = [int(d) for d in args.devid.split(",")]
    engine = chat.ChatGLM()
    engine.init(devices, args.model_path, args.tokenizer_path)

    # 4. inference
    res = {}
    subject_num = len(os.listdir(test_path))
    print(f"Subject numbers: {subject_num}")

    for dev_csv_file, test_csv_file in zip(os.listdir(dev_path), os.listdir(test_path)):
        dev_csv_path = os.path.join(dev_path, dev_csv_file)
        test_csv_path = os.path.join(test_path, test_csv_file)
        dev_df = pd.read_csv(dev_csv_path)
        test_df = pd.read_csv(test_csv_path)

        subject = test_csv_file.replace("_test.csv", "")
        subject_zh = subject_map[subject][1]
        dev_row = [dev_df.loc[i] for i in range(example_num)]

        subject_dict = {}
        for i in tqdm(range(len(test_df))):
            content = construct_content(subject_zh, dev_row, test_df.loc[i], example_num)
            option = engine.predict_option(content)
            subject_dict[str(i)] = option
        res[subject] = subject_dict

    # 5. deinit & save
    engine.deinit()
    dump_json(res, submit_path)

if __name__ == "__main__":
    main()
