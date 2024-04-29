import os
import json
import argparse
from tqdm import tqdm
import pandas as pd
from models.ChatGLM3.python_demo import chat
from transformers import AutoTokenizer
import re

def load_json(json_path):
    with open(json_path, 'r') as f:
        res = json.load(f)
    return res

def dump_json(dic, json_path):
    with open(json_path, 'w') as json_file:
        json.dump(dic, json_file)
    return

def construct_prompt(subject, dev_row, test_row, example_num):
    sys_pattern = "以下是中国关于{}考试的单项选择题，请选出其中的正确答案。\n\n"
    question_pattern = "{}\nA. {}\nB. {}\nC. {}\nD. {}\n答案：{}\n"
    test_pattern = "{}\nA. {}\nB. {}\nC. {}\nD. {}\n答案："

    res = sys_pattern.format(subject)
    for i in range(example_num):
        res = res + question_pattern.format(dev_row[i].question, dev_row[i].A, dev_row[i].B, dev_row[i].C, dev_row[i].D, dev_row[i].anwser)
    res = res + test_pattern.format(test_row.question, test_row.A, test_row.B, test_row.C, test_row.D)
    return res

def bmodel_infer(model, tokenizer, prompt, history):
    tokens = tokenizer.build_chat_input(prompt, history=history)['input_ids'].tolist()[0]
    token = model.generate(tokens, tokenizer.eos_token_id)
    answer_cur = tokenizer.decode(token)
    return answer_cur
def bmodel_infer_fast(model, tokenizer, prompt, history):
    tokens = tokenizer.build_chat_input(prompt, history=history)['input_ids'].tolist()[0]
    token = model.forward_first(tokens)
    token = model.forward_next()
    token = model.forward_next()
    answer_cur = tokenizer.decode(token)
    return answer_cur
def extract_cot_answer(self, line, gen_ans):
    m = re.findall(r'所以答案是(.+?)。', gen_ans, re.M)
    if len(m) > 0 and m[-1] in self.choices:
        return m[-1], True
    answer_patterns = [
        r'([ABCD])是正确的',
        r'选项([ABCD])正确',
        r'答案为([ABCD])',
        r'答案是([ABCD])',
        r'答案([ABCD])',
        r'选择([ABCD])',
        r'答案：([ABCD])',
        r'选择答案([ABCD])'
    ]
    # RE extraction
    for answer_pattern in answer_patterns:
        m = re.search(answer_pattern, gen_ans, re.M)
        if m:
            answer = m.group(1)
            return answer, False
    # only containing one choice-character
    m = re.findall(r'[ABCD]', gen_ans, re.M)
    if len(m) == 1:
        answer = m[0]
        return answer, False
    answer_word_counter = 0
    # only containing one choice-context
    for c in self.choices:
        if str(line[f'{c}']) in gen_ans:
            answer = c
            answer_word_counter += 1
    if answer_word_counter == 1:
        return answer, False
    return '-', False

def main(args):
    # 1. define params
    example_num = 0
    dev_path = "ceval-exam/dev"
    test_path = "ceval-exam/test"
    if "int8" in args.model_path:
        submit_path ="submisstion_int8.json"
    elif "int4" in args.model_path:
        submit_path ="submisstion_int4.json"
    elif "f16" in args.model_path:
        submit_path ="submission_f16.json"
    subject_path = "subject_mapping.json"
    subject_map = load_json(subject_path)
    
    # 2. create engine
    model = chat.ChatGLM()
    devices = [int(d) for d in args.devid.split(",")]
    model.init(devices, args.model_path)
    model.temperature = args.temperature
    model.top_p = args.top_p
    model.repeat_penalty = args.repeat_penalty
    model.repeat_last_n = args.repeat_last_n
    model.max_new_tokens = args.max_new_tokens
    model.generation_mode = args.generation_mode
    model.prompt_mode = args.prompt_mode
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # 3. inference
    res = {}
    subject_num = len(os.listdir(test_path))
    print(f"Subject numbers: {subject_num}")
    count = 0
    for dev_csv_file, test_csv_file in zip(os.listdir(dev_path), os.listdir(test_path)):
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
        for i in tqdm(range(len(test_df))):
            prompt = construct_prompt(subject_zh, dev_row, test_df.loc[i], example_num)
            print("")
            print("prompt:", prompt)
            if args.eval_mode == "fast":
                pred = bmodel_infer_fast(model, tokenizer, prompt, history = [])
            else:
                pred = bmodel_infer(model, tokenizer, prompt, history = [])
            print("prediction:", pred)
            subject_dict[str(i)] = pred
        res[subject] = subject_dict

    # 4. deinit & save
    
    dump_json(res, submit_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--model_path', type=str, help='Path to the bmodel file.')
    parser.add_argument('--tokenizer_path', type=str, help='Path to the tokenizer file.')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')
    parser.add_argument('--top_p', type=float, default=1.0, help='cumulative probability of token words to consider as a set of candidates')
    parser.add_argument('--repeat_penalty', type=float, default=1.0, help='penalty for repeated tokens')
    parser.add_argument('--repeat_last_n', type=int, default=32, help='repeat penalty for recent n tokens')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new token length to generate')
    parser.add_argument('--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
    parser.add_argument('--prompt_mode', type=str, choices=["prompted", "unprompted"], default="prompted", help='use prompt format or original input')
    parser.add_argument('--eval_mode', type=str, choices=["fast", "default"], default="default", help='eval_mode(fast or default)')
    
    args = parser.parse_args()
    main(args)
