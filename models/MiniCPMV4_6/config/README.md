---
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- minicpm-v
- multimodal
- On-Device Model
- lightweight
library_name: transformers
base_model: openbmb/MiniCPM-V-4.6
base_model_relation: quantized
---

> **This repository hosts the AWQ (W4A16, AutoAWQ) quantized version of [MiniCPM-V 4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6).** For the original BF16 weights and the full model card, please refer to [openbmb/MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6).

A Pocket-Sized MLLM for Ultra-Efficient Image and Video Understanding on Your Phone

[GitHub](https://github.com/OpenBMB/MiniCPM-o) | [CookBook](https://github.com/OpenSQZ/MiniCPM-V-CookBook) | [Demo](https://huggingface.co/spaces/openbmb/MiniCPM-V-4.6-AWQ-Demo) |
[Feishu (Lark)](https://raw.githubusercontent.com/openbmb/MiniCPM-V/main/assets/feishu_qrcode.png)

## News

* [2026.05.17] ⭐️⭐️⭐️ We release the API service of MiniCPM-V 4.6, with a **public free API key** together! Try [it](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/api.md) now.



## MiniCPM-V 4.6

**MiniCPM-V 4.6** is our most edge-deployment-friendly model to date. The model is built based on SigLIP2-400M and the Qwen3.5-0.8B LLM. It inherits the strong single-image, multi-image, and video understanding capabilities of MiniCPM-V family, while significantly improving computation efficiency. It also introduces mixed 4x/16x visual token compression. Notable features of MiniCPM-V 4.6 include:

- 🔥 **Leading Foundation Capability.**
  MiniCPM-V 4.6 scores 13 on the Artificial Analysis Intelligence Index benchmark, outperforming Qwen3.5-0.8B's score of 10 with 19x fewer token cost, and Qwen3.5-0.8B-Thinking's score of 11 with 43x fewer token cost. It also surpasses the larger Ministral 3 3B (score of 11).

- 💪 **Strong Multimodal Capability.**
  MiniCPM-V 4.6 outperforms Qwen3.5-0.8B on most vision-language understanding tasks, and reaches Qwen3.5 2B-level capability on many benchmarks including OpenCompass, RefCOCO, HallusionBench, MUIRBench, and OCRBench.
- 🚀 **Ultra-Efficient Architecture.**
  Based on the latest technique in [LLaVA-UHD v4](https://github.com/THUMAI-Lab/LLaVA-UHD-v4), MiniCPM-V 4.6 reduces the visual encoding computation FLOPs by more than 50%. It enables MiniCPM-V 4.6 to achieve better efficiency to even smaller models, achieving ~1.5x token throughput compared to Qwen3.5-0.8B. 
  It also supports mixed 4x/16x visual token compression rate, allowing flexible switching between accuracy and speed.
- 📱 **Broad Mobile Platform Coverage.**
  MiniCPM-V 4.6 can be deployed across all three mainstream mobile platforms — iOS, Android, and HarmonyOS. With every edge adaptation code open-sourced, developers can reproduce the on-device experience in [just a few steps](#deploy-minicpm-v-46-on-ios-android-and-harmonyos-platforms).
- 🛠️ **Developer Friendly.**
  MiniCPM-V 4.6 is adapted to [inference frameworks](#inference-and-training) such as vLLM, SGLang, llama.cpp, Ollama, and supports [fine-tuning ecosystems](#inference-and-training) such as SWIFT and LLaMA-Factory. Developers can quickly customize models for new domains and tasks on consumer-grade GPUs. We provide multiple quantized variants across GGUF, BNB, AWQ, and GPTQ formats.


### Evaluation <!-- omit in toc -->

**Overall Performance (Instruct)**

<p align="center">
  <img src="https://raw.githubusercontent.com/openbmb/MiniCPM-V/main/assets/minicpmv4.6/instruct.png" width="90%"></img>
</p>


<details>
<summary>Click to view MiniCPM-V 4.6-Thinking performance.</summary>


<p align="center">
  <img src="https://raw.githubusercontent.com/openbmb/MiniCPM-V/main/assets/minicpmv4.6/thinking.png" width="90%"></img>
</p>


</details>


<details>
<summary>Click to view MiniCPM-V 4.6 inference efficiency results.</summary>


**High-Concurrency Throughput**

<p align="center">
  <img src="https://raw.githubusercontent.com/openbmb/MiniCPM-V/main/assets/minicpmv4.6/throughput.png" width="60%"></img>
</p>

**Single Request TTFT (ms)**

<p align="center">
  <img src="https://raw.githubusercontent.com/openbmb/MiniCPM-V/main/assets/minicpmv4.6/ttft.png" width="60%"></img>
</p>


</details>


### Examples <!-- omit in toc -->

#### Overall

<div align="center">
  <a href="https://www.youtube.com/watch?v=Ch5UG1FoysM"><img src="https://raw.githubusercontent.com/openbmb/MiniCPM-V/main/assets/minicpmv4.6/video_play.png" width="70%"></a>
</div>

MiniCPM-V 4.6 can be deployed across three mainstream end-side platforms — **iOS, Android and HarmonyOS**. The clips below are raw screen recordings on phone devices without edition.

<table align="center">
  <tr>
    <td align="center"><b>iPhone</b><br><sub>iPhone 17 Pro Max</sub></td>
    <td align="center"><b>Android</b><br><sub>Redmi K70</sub></td>
    <td align="center"><b>HarmonyOS</b><br><sub>HUAWEI nova 14</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/openbmb/MiniCPM-V/main/assets/minicpmv4.6/v46_iphone_en_handwriting.gif" width="100%"/></td>
    <td align="center"><img src="https://raw.githubusercontent.com/openbmb/MiniCPM-V/main/assets/minicpmv4.6/v46_android_en_refraction.gif" width="100%"/></td>
    <td align="center"><img src="https://raw.githubusercontent.com/openbmb/MiniCPM-V/main/assets/minicpmv4.6/v46_harmonyos_en_ticket.gif" width="100%"/></td>
  </tr>
</table>


### Usages

#### Inference with Transformers <!-- omit in toc -->
##### Installation <!-- omit in toc -->

```bash
pip install "transformers[torch]>=5.7.0" torchvision torchcodec
```

> **Note on CUDA compatibility:** `torchcodec` (used for video decoding) may have compatibility issues with certain CUDA versions. For example, `torch>=2.11` bundles CUDA 13.1 by default, while environments with CUDA 12.x may encounter errors such as `RuntimeError: Could not load libtorchcodec`. Two workarounds:
>
> 1. **Replace `torchcodec` with `PyAV`** — supports both image and video inference without CUDA version constraints:
>    ```bash
>    pip install "transformers[torch]>=5.7.0" torchvision av
>    ```
> 2. **Pin the CUDA version** when installing torch to match your environment (e.g. CUDA 12.8):
>    ```bash
>    pip install "transformers>=5.7.0" torchvision torchcodec --index-url https://download.pytorch.org/whl/cu128
>    ```

##### Load Model <!-- omit in toc -->

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "openbmb/MiniCPM-V-4.6-AWQ"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

# Flash Attention 2 is recommended for better acceleration and memory saving,
# especially in multi-image and video scenarios.
# model = AutoModelForImageTextToText.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )
```

##### Image Inference <!-- omit in toc -->

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/refract.png"},
            {"type": "text", "text": "What causes this phenomenon?"},
        ],
    }
]

downsample_mode = "16x"  # Using `downsample_mode="4x"` for Finer Detail

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
    downsample_mode=downsample_mode,
    max_slice_nums=36,
).to(model.device)

generated_ids = model.generate(**inputs, downsample_mode=downsample_mode, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
```

##### Video Inference <!-- omit in toc -->

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "url": "https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/football.mp4"},
            {"type": "text", "text": "Describe this video in detail. Follow the timeline and focus on on-screen text, interface changes, main actions, and scene changes."},
        ],
    }
]

downsample_mode = "16x"  # Using `downsample_mode="4x"` for Finer Detail

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
    downsample_mode=downsample_mode,
    max_num_frames=128,
    stack_frames=1,
    max_slice_nums=1,
    use_image_id=False,
).to(model.device)

generated_ids = model.generate(**inputs, downsample_mode=downsample_mode, max_new_tokens=2048)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
```

##### Advanced Parameters <!-- omit in toc -->

You can customize image/video processing by passing additional parameters to `apply_chat_template`:

| Parameter | Default | Applies to | Description |
|-----------|---------|------------|-------------|
| `downsample_mode` | `"16x"` | Image & Video | Visual token downsampling. `"16x"` merges tokens for efficiency; `"4x"` keeps 4× more tokens for finer detail. Must also be passed to `generate()`. |
| `max_slice_nums` | `9` | Image & Video | Maximum number of slices when splitting a high-resolution image. Higher values preserve more detail for large images. Recommended: `36` for image, `1` for video. |
| `max_num_frames` | `128` | Video only | The `max_num_frames` parameter dynamically controls the temporal context length and prevents VRAM overflow: <br> **Short Videos** (duration ≤ `max_num_frames` sec): The processor defaults to **1 FPS**, capturing second-by-second details without hitting the upper limit. <br> **Long Videos** (duration > `max_num_frames` sec): The processor automatically switches to **uniform sampling**, selecting exactly `max_num_frames` evenly spaced across the entire timeline. |
| `stack_frames` | `1` | Video only | Total sample points per second. `1` = main frame only (no stacking). `N` (N>1) = 1 main frame + N−1 sub-frames per second; the sub-frames are composited into a grid image and interleaved with main frames. Recommended setting is `1` for short videos, and `3` or `5` for long videos. |
| `use_image_id` | `True` | Image & Video | Whether to prepend `<image_id>N</image_id>` tags before each image/frame placeholder. Set `True` for image, `False` for video. |

> **Note:** `downsample_mode` must be passed to **both** `apply_chat_template` (for correct placeholder count) and `generate` (for the vision encoder). All other parameters only need to be passed to `apply_chat_template`.

##### Serving with `transformers serve` <!-- omit in toc -->

Hugging Face Transformers includes a lightweight OpenAI-compatible server for quick testing and moderate-load deployment.

```bash
pip install "transformers[serving]>=5.7.0"
```

Start the server:

```bash
transformers serve openbmb/MiniCPM-V-4.6-AWQ --port 8000 --host 0.0.0.0 --continuous-batching
```

Send a request:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openbmb/MiniCPM-V-4.6-AWQ",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/refract.png"}},
        {"type": "text", "text": "What causes this phenomenon?"}
      ]
    }]
  }'
```

Tool calling example:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "openbmb/MiniCPM-V-4.6-AWQ",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "the weather of Beijing"}
  ]}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
      }
    }
  }]
}'
```

The model returns a natural-language explanation followed by a structured <tool_call> block embedded in the content field. Note that a dedicated tool call parser for this format has not yet been added to the transformers library, so the tool calls need to be extracted manually via regex for now.

```
{
    "id": "f4f09c7d-8045-4cb1-ade9-07aa5dee637d",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "I need to check the current weather for Beijing, so I will call the get_weather function.\n\n<tool_call>\n<function=get_weather>\n<parameter=location>\nBeijing\n</parameter>\n</function>\n</tool_call>",
                "role": "assistant"
            }
        }
    ],
    "created": 1778748859,
    "model": "openbmb/MiniCPM-V-4.6-AWQ@main",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 47,
        "prompt_tokens": 283,
        "total_tokens": 330
    }
}
```

#### Handling Escaped Newlines in Model Outputs <!-- omit in toc -->

In some cases, the model might output escaped newline characters `\n` as string literals instead of actual newlines. To render the text correctly, especially in UI layers, you can use the following utility function. This function carefully replaces literal `\n` with real newlines while protecting scenarios where `\n` has specific semantic meaning.

**Utility Function:**

```python
import re

_PATTERN = re.compile(
    r'(```[\s\S]*?```'       # fenced code blocks
    r'|`[^`]+`'              # inline code
    r'|\$\$[\s\S]*?\$\$'     # display math
    r'|\$[^$]+\$'            # inline math
    r'|\\\([\s\S]*?\\\)'     # \(...\)
    r'|\\\[[\s\S]*?\\\]'     # \[...\]
    r')'
    r'|(?<!\\)(?:\\r\\n|\\[nr])'
)

def normalize_response_text(text: str) -> str:
    """
    Lightweight post-processing: Converts literal '\\n' to actual newlines, 
    while protecting code blocks, inline code, and LaTeX commands.
    """
    if not isinstance(text, str) or "\\" not in text:
        return text
    return _PATTERN.sub(lambda m: m.group(1) or '\n', text)
```

#### Deploy MiniCPM-V 4.6 on iOS, Android, and HarmonyOS Platforms <!-- omit in toc -->

We have adapted MiniCPM-V 4.6 for deployment on **iOS, Android, and HarmonyOS** platforms, with **all edge adaptation code fully open-sourced**. Developers can reproduce the on-device experience in just a few steps. Visit our [edge deployment repository](https://github.com/OpenBMB/MiniCPM-V-edge-demo) for platform-specific build guides, or go to the [download page](https://github.com/OpenBMB/MiniCPM-V-edge-demo/blob/main/DOWNLOAD.md) to try pre-built apps directly.

<a id="inference-and-training"></a>
#### Use MiniCPM-V 4.6 in Other Inference and Training Frameworks <!-- omit in toc -->

MiniCPM-V 4.6 supports multiple inference and training frameworks. Below are quick-start commands for each. For full details, see our [Cookbook](https://github.com/OpenSQZ/MiniCPM-V-CookBook).

<details>
<summary><b>vLLM</b> — <a href="https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/vllm/minicpm-v4_6_vllm.md">Full Guide</a></summary>

```bash
vllm serve openbmb/MiniCPM-V-4.6-AWQ \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

> **Note:** `--enable-auto-tool-choice` and `--tool-call-parser qwen3_coder` enable tool/function calling support. If you don't need tool use, you can omit these flags and simply run `vllm serve openbmb/MiniCPM-V-4.6-AWQ`.

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "openbmb/MiniCPM-V-4.6-AWQ",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/refract.png"}},
    {"type": "text", "text": "What causes this phenomenon?"}
  ]}]
}'
```


Tool calling example:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "openbmb/MiniCPM-V-4.6-AWQ",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "北京的天气"}
  ]}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
      }
    }
  }]
}'
```

</details>

<details>
<summary><b>SGLang</b> — <a href="https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/sglang/minicpm-v4_6_sglang.md">Full Guide</a></summary>

```bash
python -m sglang.launch_server --model openbmb/MiniCPM-V-4.6-AWQ --port 30000
```

```bash
curl -s http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "openbmb/MiniCPM-V-4.6-AWQ",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/refract.png"}},
    {"type": "text", "text": "What causes this phenomenon?"}
  ]}]
}'
```

</details>

<details>
<summary><b>llama.cpp</b> — <a href="https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp/minicpm-v4_6_llamacpp.md">Full Guide</a></summary>

```bash
llama-server -m MiniCPM-V-4.6-Q4_K_M.gguf --port 8080
```

```bash
curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "MiniCPM-V-4.6",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/openbmb/DemoCase/resolve/main/refract.png"}},
    {"type": "text", "text": "What causes this phenomenon?"}
  ]}]
}'
```

</details>

<details>
<summary><b>Ollama</b> — <a href="https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/ollama/minicpm-v4_6_ollama.md">Full Guide</a></summary>

```bash
ollama run minicpm-v-4.6
```

In the interactive session, paste an image path or URL directly to chat with the model.

</details>

<details>
<summary><b>LLaMA-Factory</b> (Fine-tuning) — <a href="https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/finetune/llamafactory_minicpmv46.md">Full Guide</a></summary>

```bash
llamafactory-cli train examples/train_lora/minicpmv4_6_lora_sft.yaml
```

</details>

<details>
<summary><b>ms-swift</b> (Fine-tuning) — <a href="https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/finetune/swift_minicpmv46.md">Full Guide</a></summary>

```bash
swift sft --model_type minicpm-v-4_6 --dataset <your-dataset>
```

</details>

## License

#### Model License
* The MiniCPM-o/V model weights and code are open-sourced under the [Apache-2.0](https://github.com/OpenBMB/MiniCPM-V/blob/main/LICENSE) license.

#### Statement
* As MLLMs, MiniCPM-o/V models generate content by learning a large number of multimodal corpora, but they cannot comprehend, express personal opinions, or make value judgements. Anything generated by MiniCPM-o/V models does not represent the views and positions of the model developers
* We will not be liable for any problems arising from the use of MiniCPM-o/V models, including but not limited to data security issues, risk of public opinion, or any risks and problems arising from the misdirection, misuse, dissemination, or misuse of the model.


## Technical Reports and Key Techniques Papers

👏 Welcome to explore key techniques of MiniCPM-o/V and other multimodal projects of our team:

**Technical Reports:** [MiniCPM-o 4.5](https://huggingface.co/papers/2604.27393) | [MiniCPM-V 4.5](https://arxiv.org/abs/2509.18154) | [MiniCPM-o 2.6](https://openbmb.notion.site/MiniCPM-o-2-6-A-GPT-4o-Level-MLLM-for-Vision-Speech-and-Multimodal-Live-Streaming-on-Your-Phone-185ede1b7a558042b5d5e45e6b237da9) | [MiniCPM-Llama3-V 2.5](https://arxiv.org/abs/2408.01800) | [MiniCPM-V 2.0](https://openbmb.vercel.app/minicpm-v-2)

**Other Multimodal Projects:** [VisCPM](https://github.com/OpenBMB/VisCPM/tree/main) | [RLPR](https://github.com/OpenBMB/RLPR) | [RLHF-V](https://github.com/RLHF-V/RLHF-V) | [LLaVA-UHD](https://github.com/thunlp/LLaVA-UHD) | [RLAIF-V](https://github.com/RLHF-V/RLAIF-V) | [LLaVA-UHD-v4](https://arxiv.org/abs/2605.08985 )


## Citation <!-- omit in toc -->

If you find our model/code/paper helpful, please consider citing our papers 📝 and staring us ⭐️！

```bib
@proceedings{yu2025minicpmv45cookingefficient,
      title={MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipe}, 
      author={Tianyu Yu and Zefan Wang and Chongyi Wang and Fuwei Huang and Wenshuo Ma and Zhihui He and Tianchi Cai and Weize Chen and Yuxiang Huang and Yuanqian Zhao and others},
      year={2025},
      url={https://arxiv.org/abs/2509.18154}, 
}

@article{yao2024minicpm,
  title={MiniCPM-V: A GPT-4V Level MLLM on Your Phone},
  author={Yao, Yuan and Yu, Tianyu and Zhang, Ao and Wang, Chongyi and Cui, Junbo and Zhu, Hongji and Cai, Tianchi and Li, Haoyu and Zhao, Weilin and He, Zhihui and others},
  journal={arXiv preprint arXiv:2408.01800},
  year={2024}
}
```