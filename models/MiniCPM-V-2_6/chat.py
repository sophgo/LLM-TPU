import torch
import os
from PIL import Image
from transformers import AutoModel, AutoTokenizer

PWD = os.getcwd().replace('\\','/')
MiniCPMV_PATH = "{}/../../../MiniCPM-V-2_6".format(PWD)

model = AutoModel.from_pretrained(MiniCPMV_PATH, trust_remote_code=True,
    attn_implementation='eager', torch_dtype=torch.bfloat16, device_map="cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(MiniCPMV_PATH, trust_remote_code=True)

image = Image.open('./python_demo/dog.jpg').convert('RGB')
question = 'What is in the image?'
msgs = [{'role': 'user', 'content': [image, question]}]

res = model.chat(image=None,msgs=msgs,tokenizer=tokenizer)
print(res)
