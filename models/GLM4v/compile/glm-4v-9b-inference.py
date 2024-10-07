import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained("./glm-4v-9b", trust_remote_code=True)

query = '描述这张图片'
image = Image.open("./glm-4v-9b/demo_small.jpeg").convert('RGB')
inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)  # chat mode

inputs = inputs.to(device)
model = AutoModelForCausalLM.from_pretrained(
    "./glm-4v-9b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    print('begin inference')
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0]))