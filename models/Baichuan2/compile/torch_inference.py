import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model_path', help='下载模型的绝对路径')
args = parser.parse_args()
model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float32, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(model_path)
messages = []
messages.append({"role": "user", "content": "解释一下“温故而知新”"})
response = model.chat(tokenizer, messages)
print(response)