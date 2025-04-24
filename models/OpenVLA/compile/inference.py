# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

# Load Processor & VLA
model_path = "/workspace/models/openvla-7b"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    model_path, 
    attn_implementation="eager",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.float, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cpu")

# Grab image input & format prompt
# image: Image.Image = get_from_camera(...)
url = "./codalm3.png"
image = Image.open(str(url)).convert("RGB")
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cpu", dtype=torch.float)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
breakpoint()

