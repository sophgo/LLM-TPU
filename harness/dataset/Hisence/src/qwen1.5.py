import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from mp_utils import choices, format_example, gen_prompt, softmax, run_eval, run_subject_eval

from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer


def eval_chat(model, tokenizer, subject, dev_df, test_df, num_few_shot, max_length, cot, device):
    cors = []
    all_preds = []
    answers = choices[: test_df.shape[1] - 2]

    for i in tqdm(range(test_df.shape[0])):
        prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
        prompt = gen_prompt(dev_df=dev_df,
                            subject=subject,
                            prompt_end=prompt_end,
                            num_few_shot=num_few_shot,
                            tokenizer=tokenizer,
                            max_length=max_length,
                            cot=cot)
        label = test_df.iloc[i, test_df.shape[1] - 1]

        #根据prompt推理结果
        # prompt = "你好"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        if device == "cuda":
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                temperature=0.5
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
        elif device == "tpu":
            model_inputs = tokenizer([text])
            generated_ids = model.generate(
                model_inputs.input_ids[0],
                tokenizer.eos_token_id
            )
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response)

        if response and response[0] in choices:
            cors.append(response[0] == label)
        all_preds.append(response.replace("\n", ""))

    acc = np.mean(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    print("{} results, {} inappropriate formated answers.".format(len(cors), len(all_preds)-len(cors)))
    return acc, all_preds, None

all = [
    "agronomy",
    "anatomy",
    "ancient_chinese",
    "arts",
    "astronomy",
    "business_ethics",
    "chinese_civil_service_exam",
    "chinese_driving_rule",
    "chinese_food_culture",
    "chinese_foreign_policy",
    "chinese_history",
    "chinese_literature",
    "chinese_teacher_qualification",
    "clinical_knowledge",
    "college_actuarial_science",
    "college_education",
    "college_engineering_hydrology",
    "college_law",
    "college_mathematics",
    "college_medical_statistics",
    "college_medicine",
    "computer_science",
    "computer_security",
    "conceptual_physics",
    "construction_project_management",
    "economics",
    "education",
    "electrical_engineering",
    "elementary_chinese",
    "elementary_commonsense",
    "elementary_information_and_technology",
    "elementary_mathematics",
    "ethnology",
    "food_science",
    "genetics",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_geography",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_politics",
    "human_sexuality",
    "international_law",
    "journalism",
    "jurisprudence",
    "legal_and_moral_basis",
    "logical",
    "machine_learning",
    "management",
    "marketing",
    "marxist_theory",
    "modern_chinese",
    "nutrition",
    "philosophy",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_study",
    "sociology",
    "sports_science",
    "traditional_chinese_medicine",
    "virology",
    "world_history",
    "world_religions"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--lora_weights", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="results/ChatGLM-6B")
    parser.add_argument("--num_few_shot", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--load_in_8bit", action='store_true')
    parser.add_argument("--subjects", type=str, nargs='+', default= all) #['high_school_geography','electrical_engineering'])
    parser.add_argument("--cot", action='store_true')
    parser.add_argument("--device", type=str, choices=["cuda", "tpu"], default="cuda")
    parser.add_argument('--model_path', type=str, required=True, help='path to the bmodel file')
    parser.add_argument('--top_p', type=float, default=1.0, help='cumulative probability of token words to consider as a set of candidates')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new token length to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')
    parser.add_argument("--devid", type=str, default='0')
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument('--generation_mode', type=str, default="greedy", help='mode for generating next token.')
    parser.add_argument('--prompt_mode', type=str, choices=["prompted", "unprompted"], default="prompted", help='use prompt format or original input')
    args = parser.parse_args()

    # Initialize models
    if args.device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype="auto",
            device_map="auto"
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    elif args.device == "tpu":
        from Qwen1_5.python_demo import chat
        devices = [int(d) for d in args.devid.split(",")]
        model = chat.Qwen()
        model.init(
            devices,
            args.model_path,
            args.temperature,
            args.top_p,
            args.max_new_tokens,
            args.generation_mode,
            args.prompt_mode,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    print("subject:", args.subjects)
    # Always use Chat-style evaluation
    # run_eval(model, tokenizer, eval_chat, args)
    run_subject_eval(model, tokenizer, eval_chat, args)


