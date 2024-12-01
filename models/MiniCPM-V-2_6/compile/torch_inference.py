import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('./MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='eager', torch_dtype=torch.float) # sdpa or flash_attention_2, no eager
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained('./MiniCPM-V-2_6', trust_remote_code=True)

image = Image.open('../python_demo/test0.jpg').convert('RGB')
question = '请详细描述一下图片内容'
msgs = [{'role': 'user', 'content': [image, question]}]


## if you want to use streaming, please make sure sampling=True and stream=True
## the model.chat will return a generator
res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=False,
    do_sample=False,
    num_beams=1,
)

generated_text = ""
for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end='')
