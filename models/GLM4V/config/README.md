---
license: mit
---

# GLM-4.1V-9B-Thinking

<div align="center">
<img src=https://raw.githubusercontent.com/THUDM/GLM-4.1V-Thinking/99c5eb6563236f0ff43605d91d107544da9863b2/resources/logo.svg width="40%"/>
</div>
<p align="center">
    📖 Read the GLM-4.1V-9B-Thinking <a href="https://arxiv.org/abs/2507.01006" target="_blank">paper</a>.
    <br>
    💡 Try GLM-4.1V-9B-Thinking online now on <a href="https://huggingface.co/spaces/THUDM/GLM-4.1V-9B-Thinking-Demo" target="_blank">Hugging Face</a> or <a href="https://modelscope.cn/studios/ZhipuAI/GLM-4.1V-9B-Thinking-Demo" target="_blank">ModelScope</a>.
    <br>
    📍 Use the GLM-4.1V-9B-Thinking API service on the <a href="https://www.bigmodel.cn/dev/api/visual-reasoning-model/GLM-4.1V-Thinking">Zhipu BigModel Open Platform</a>.
</p>

## Model Introduction

Vision-language models (VLMs) have become a key cornerstone of intelligent systems. As real-world intelligent tasks become increasingly complex, VLMs urgently need to go beyond basic multimodal perception and gradually strengthen their reasoning capabilities in complex tasks, improving their accuracy, comprehensiveness, and intelligence, thereby enabling intelligent tasks such as complex problem solving, long-context understanding, and multimodal agents.

Based on the [GLM-4-9B-0414](https://github.com/THUDM/GLM-4) base model, we introduce the new open-source VLM **GLM-4.1V-9B-Thinking**, which incorporates a thinking paradigm and comprehensively improves model capabilities through Reinforcement Learning with Curriculum Sampling (RLCS), achieving the strongest performance among vision-language models at the 10B parameter level, and matching or even surpassing Qwen-2.5-VL-72B, which has 8x the parameters, on 18 benchmark tasks. We also open-source the base model **GLM-4.1V-9B-Base**, hoping to help more researchers explore the capability boundaries of vision-language models.

![rl](https://raw.githubusercontent.com/THUDM/GLM-4.1V-Thinking/refs/heads/main/resources/rl.jpeg)

Compared with the previous-generation CogVLM2 and GLM-4V series models, **GLM-4.1V-Thinking** has the following improvements:

1. The first reasoning model in the series, achieving world-leading levels not only in mathematics but also across multiple sub-domains.
2. Supports **64k** context length.
3. Supports **arbitrary aspect ratios** and image resolutions up to **4k**.
4. Provides an open-source model version supporting **both Chinese and English**.

## Benchmark Information

By introducing the Chain-of-Thought reasoning mechanism, GLM-4.1V-9B-Thinking comprehensively surpasses traditional non-reasoning vision models in answer accuracy, content richness, and interpretability. It achieves the best results among 10B-level models on 23 of 28 evaluation tasks, and even surpasses Qwen-2.5-VL-72B, which has 8x the parameters, on 18 tasks.

![bench](https://raw.githubusercontent.com/THUDM/GLM-4.1V-Thinking/refs/heads/main/resources/bench.jpeg)

## Quick Inference

Here is an example of single-image inference using `transformers`. First, install the `transformers` library from source.
```
pip install git+https://github.com/huggingface/transformers.git
```

Then run the following code:

```python
from transformers import AutoProcessor, Glm4vForConditionalGeneration
import torch

MODEL_PATH = "THUDM/GLM-4.1V-9B-Thinking"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://model-demo.oss-cn-hangzhou.aliyuncs.com/Grayscale_8bits_palette_sample_image.png"
            },
            {
                "type": "text",
                "text": "describe this image"
            }
        ],
    }
]
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = Glm4vForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=8192)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print(output_text)
```


For more code, such as video inference and web demo deployment, please check our [github](https://github.com/THUDM/GLM-4.1V-Thinking).
