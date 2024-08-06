import os
import re
import sys
import json
import torch


def load_json(json_path):
    with open(json_path, "r") as f:
        res = json.load(f)
    return res

def dump_json(dic, json_path):
    with open(json_path, "w") as json_file:
        json.dump(dic, json_file)
    return

def record_time(cost_time, cost_time_path):
    print(cost_time)
    total_cost_time = 0
    with open(cost_time_path, "w") as file:
        for key, value in cost_time.items():
            file.write(f"{key}: {value}\n")
            total_cost_time += value
    print(f"Total cost time: {total_cost_time}")

def construct_prompt(subject, dev_row, test_row, example_num):
    sys_pattern = "以下是中国关于{}考试的单项选择题，请选出其中的正确答案。\n\n"
    question_pattern = "{}\nA. {}\nB. {}\nC. {}\nD. {}\n答案：{}\n"
    test_pattern = "{}\nA. {}\nB. {}\nC. {}\nD. {}\n答案："

    res = sys_pattern.format(subject)
    for i in range(example_num):
        res = res + question_pattern.format(
            dev_row[i].question,
            dev_row[i].A,
            dev_row[i].B,
            dev_row[i].C,
            dev_row[i].D,
            dev_row[i].anwser,
        )
    res = res + test_pattern.format(
        test_row.question, test_row.A, test_row.B, test_row.C, test_row.D
    )

    print("")
    print("prompt:", res)
    return res

def extract_cot_answer(gen_ans):
    choices = ["A", "B", "C", "D"]
    answer_patterns = [
        r"([ABCD])是正确",
        r"([ABCD])正确",
        r"答案.([ABCD])",
        r"答案([ABCD])",
        r"选(?:选项)?([ABCD])",
        r"选择(?:选项)?([ABCD])",
        r"([ABCD])是对的",
    ]
    # RE extraction
    for answer_pattern in answer_patterns:
        m = re.search(answer_pattern, gen_ans, re.M)
        if m:
            answer = m.group(1)
            return answer

    m = re.findall(r"[ABCD]", gen_ans, re.M)
    if len(m) == 1:
        answer = m[0]
        return answer
    elif gen_ans[0] in choices:
        return gen_ans[0]
    elif len(m) > 1:
        return m[-1]
    return "-"

def load_model(args):
    if args.device == "cuda":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype="auto", device_map="auto"
        ).eval()
    elif args.device == "tpu":
        sys.path.append("../../")
        try:
            if args.model_name == "ChatGLM3":
                from models.ChatGLM3.python_demo import chat
            elif args.model_name == "Qwen2":
                from models.Qwen2.python_demo import chat
        except:
            module_path = f"../../models/{args.model_name}/python_demo"
            raise ImportError(
                f"No such module in {os.path.abspath(module_path)}, You need to make chat.cpython.so, please refer to C-Eval/README.md"
            )
        devices = [int(d) for d in args.devid.split(",")]
        model = chat.Qwen()
        model.init(devices, args.model_path)
        model.temperature = args.temperature
        model.top_p = args.top_p
        model.repeat_penalty = args.repeat_penalty
        model.repeat_last_n = args.repeat_last_n
        model.max_new_tokens = args.max_new_tokens
        model.generation_mode = args.generation_mode
    return model

def encode_tokens(prompt, tokenizer, args):
    messages = [
        {
            "role": "system",
            "content": "You will provide correct answer to the question.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text

def generate_and_decode_tokens(model, tokenizer, text, args):
    # generate tokens
    tok_num = args.max_new_tokens
    if args.device == "cuda":
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            model_inputs.input_ids, max_new_tokens=tok_num, num_beams=1, do_sample=False
        ).cpu().numpy()[0][-tok_num:]
    elif args.device == "tpu":
        assert(model.generation_mode == "greedy")
        assert(model.max_new_tokens == tok_num)
        model_inputs = tokenizer([text])
        generated_ids = model.generate(model_inputs.input_ids[0], tokenizer.eos_token_id)
    
    # decode tokens
    answer_cur = tokenizer.decode(generated_ids)
    return answer_cur

def inference_model(model, tokenizer, prompt, args):
    text = encode_tokens(prompt, tokenizer, args)
    answer_cur = generate_and_decode_tokens(model, tokenizer, text, args)
    answer_cur = extract_cot_answer(answer_cur)
    return answer_cur
