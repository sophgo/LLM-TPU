<div align="center">

<img src="./assets/sophgo_chip.png" alt="SOPHGO" width="720"/>

# LLM-TPU

**One-click deployment of mainstream LLMs and multimodal models on SOPHGO TPU**

*Deploy LLMs & VLMs on SOPHGO BM1684X / BM1688 / CV186X with a single command*

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-x86%20%7C%20aarch64-pink.svg)]()
[![Chip](https://img.shields.io/badge/chip-BM1684X%20%7C%20BM1688%20%7C%20CV186X-orange.svg)](https://www.sophgo.com/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](./LICENSE)
[![Contributors](https://img.shields.io/github/contributors/sophgo/LLM-TPU?color=9ea)](https://github.com/sophgo/LLM-TPU/graphs/contributors)
[![Issues](https://img.shields.io/github/issues/sophgo/LLM-TPU?color=9cc)](https://github.com/sophgo/LLM-TPU/issues)
[![Stars](https://img.shields.io/github/stars/sophgo/LLM-TPU?style=social)](https://github.com/sophgo/LLM-TPU/stargazers)

**English** · [简体中文](./README_cn.md)

[Quick Start](#-quick-start) ·
[Supported Models](#-supported-models) ·
[Compilation Flow](#-llm-compilation-flow) ·
[Advanced Features](#-advanced-features) ·
[FAQ](./docs/FAQ.md) ·
[Website](https://www.sophgo.com/)

</div>

---

## 📰 Latest News

| Date | Updates |
| :--- | :--- |
| 🔥 **2026.07.16** | **Falcon-Perception** now supports BM1684X, Python demo, referring segmentation (box + mask) → [Details](./models/Falcon-Perception/) |
| 🔥 **2026.07.09** | **LocateAnything-3B** now supports BM1684X / BM1688, Python demo, visual grounding (box / point) → [Details](./models/LocateAnything/) |
| **2026.06.30** | **MiniCPM-V-4.6** now supports BM1684X / BM1688, Python demo, image & video support → [Details](./models/MiniCPMV4_6/) |
| **2026.05.21** | **Gemma4** now supports BM1684X / BM1688, Python demo, image / video / audio support → [Details](./models/Gemma4/) |
| **2026.04.15** | **Qwen3.5** now supports BM1684X / BM1688, Python & C++ demos, image & video support → [Details](./models/Qwen3_5/) |
| **2025.10.15** | **Qwen3-VL** now supports BM1684X / BM1688, Python / C++ demos, image & video support → [Details](./models/Qwen3_VL/) |
| **2025.05.22** | **InternVL3** now supports BM1684X / BM1688, image & video support → [Details](./models/InternVL3/) |
| **2025.04.30** | **Qwen2.5-VL** now supports BM1684X / BM1688, Python / C++ demos → [Details](./models/Qwen2_5_VL/) |
| **2025.04.29** | Reasoning model **Qwen3** now supports BM1684X / BM1688 → [Details](./models/Qwen3/) |
| **2025.03.07** | **QwQ-32B** and **DeepSeek-R1-Distill-Qwen-32B** multi-chip demos adapted → [Details](./models/Qwen2_5/) |
| **2025.02.05** | Adapted **DeepSeek-R1-Distill-Qwen** series (1.5B / 7B / 14B) → [Details](./models/Qwen2_5/) |

---

## 📖 Introduction

**LLM-TPU** is an open-source project officially maintained by [SOPHGO](https://www.sophgo.com/), dedicated to deploying mainstream **generative AI models** (LLM / VLM) on SOPHGO **BM1684X / BM1688 / CV186X** series TPU chips.

```
   ┌──────────────┐    tpu-mlir    ┌──────────────┐    tpu-runtime   ┌──────────────────┐
   │  HuggingFace │ ─────────────► │   bmodel     │ ───────────────► │  PCIE / SoC       │
   │   weights    │   llm_convert  │ (quantized)  │    Python / C++  │  BM1684X / 1688  │
   └──────────────┘                └──────────────┘                  └──────────────────┘
```

- 🚀 **One-click compilation**: `llm_convert.py` exports HuggingFace weights directly to bmodel
- 🧩 **Rich model coverage**: Dozens of models including Qwen / Llama / DeepSeek / InternVL / MiniCPM / Phi / ChatGLM
- 🎯 **Multimodal**: Inference across text, image, video, and audio
- ⚡ **Efficient inference**: AWQ/GPTQ quantized models, dynamic compilation, KV Cache, multi-chip parallelism
- 🛠️ **Dual-language demos**: Popular models ship with both Python and C++ reference implementations
- 📦 **Ready to use**: Pre-compiled bmodels available for download — no compilation required

> Compiling models requires the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) environment (Docker or source build both work). Alternatively, use the pre-compiled bmodels provided in each demo. See [`models/`](./models) for the full model list.

---

## 🚀 Quick Start

Get an LLM running on your TPU device in just two steps:

```bash
git clone https://github.com/sophgo/LLM-TPU.git
cd LLM-TPU
./run.sh --model qwen2.5vl
```

### One-command demo models

| Model          | Command                           |
| :------------- | :-------------------------------- |
| Qwen3-4B       | `./run.sh --model qwen3`          |
| Qwen2.5-VL-3B  | `./run.sh --model qwen2.5vl`      |
| InternVL3-2B   | `./run.sh --model internvl3`      |

📘 For detailed steps, see **[Quick Start](./docs/Quick_Start.md)**.

<div align="center">
  <img src="./assets/test.jpg" width="45%"/>
  <img src="./assets/qwen2_5-vl.png" width="45%"/>
</div>

---

## 🧠 Supported Models

### Multimodal Models (VLM / Audio / Vision)

| Model | Supported Chips | One-click Compile | Notes |
| :--- | :---: | :---: | :--- |
| [Falcon-Perception](https://huggingface.co/tiiuae/falcon-perception) | BM1684X | — | Python, referring segmentation box + mask |
| [LocateAnything-3B](https://huggingface.co/NVIDIA/LocateAnything-3B) | BM1684X / 1688 | — | Python, visual grounding box / point |
| [Qwen3.5](https://www.modelscope.cn/collections/Qwen/Qwen35) | BM1684X / 1688 | ✅ | Python + C++, image / video |
| [Qwen3-VL](https://www.modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct) | BM1684X / 1688 | ✅ | Python + C++, image / video |
| [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ) | BM1684X / 1688 | ✅ | Python + C++ |
| [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-AWQ) | BM1684X / 1688 | ✅ | — |
| [InternVL3](https://huggingface.co/OpenGVLab/InternVL3-2B-AWQ) | BM1684X / 1688 | ✅ | Video supported |
| [Gemma4](https://huggingface.co/google/gemma-4-E2B-it) | BM1684X / 1688 | ✅ | Python, image / video / audio |
| [Gemma3](https://huggingface.co/google/gemma-3-4b-it) | BM1684X / 1688 | ✅ | — |
| Qwen-VL / InternVL2 / MiniCPM-V-2.6 / Llama3.2-Vision | BM1684X / 1688 | — | Deployed |

### LLM Models

| Family | Representative Models | One-click Compile |
| :--- | :--- | :---: |
| **Qwen** | Qwen1.5 / Qwen2 / Qwen2.5 / [Qwen3](https://huggingface.co/Qwen/Qwen3-4B-AWQ) / [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B-AWQ) | ✅ |
| **DeepSeek** | [DeepSeek-R1-Distill-Qwen](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) (1.5B / 7B / 14B / 32B) | ✅ |
| **Llama** | [Llama2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) / [Llama3](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | ✅ |
| **MiniCPM** | [MiniCPM4](https://huggingface.co/openbmb/MiniCPM4-0.5B-QAT-Int4-GPTQ-format) | ✅ |
| **Phi** | [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) / [Phi-4](https://huggingface.co/microsoft/Phi-4-mini-instruct) | ✅ |
| **ChatGLM** | [ChatGLM3](https://huggingface.co/THUDM/chatglm3-6b) / ChatGLM4 | ✅ |
| **Others** | Baichuan2 · CodeFuse · Falcon · Gemma / Gemma2 · Mistral · WizardCoder · Yi · Yi34B · LWM-Text-Chat · Megrez · MiniCPM3 · DeepSeek-V2 | — |

### Full Directory Index

The [`models/`](./models) directory currently contains the following model implementations:

**LLM**:
[Baichuan2](./models/Baichuan2) ·
[ChatGLM2](./models/ChatGLM2) ·
[ChatGLM3](./models/ChatGLM3) ·
[CodeFuse](./models/CodeFuse) ·
[DeepSeek-V2](./models/DeepSeek-V2) ·
[GLM4](./models/GLM4) ·
[Llama2](./models/Llama2) ·
[Llama3](./models/Llama3) ·
[LWM](./models/LWM) ·
[Megrez](./models/Megrez) ·
[MiniCPM3](./models/MiniCPM3) ·
[MiniCPM4](./models/MiniCPM4) ·
[Mistral](./models/Mistral) ·
[Phi-3](./models/Phi-3) ·
[Qwen](./models/Qwen) ·
[Qwen1_5](./models/Qwen1_5) ·
[Qwen2](./models/Qwen2) ·
[Qwen2_5](./models/Qwen2_5) ·
[Qwen3](./models/Qwen3) ·
[RWKV6](./models/RWKV6) ·
[RWKV7](./models/RWKV7) ·
[WizardCoder](./models/WizardCoder) ·
[Yi](./models/Yi) ·
[Yi34B](./models/Yi34B)

**Multimodal (Vision / Video / Audio)**:
[DriveMM](./models/DriveMM) ·
[Falcon-Perception](./models/Falcon-Perception) ·
[Gemma3](./models/Gemma3) ·
[Gemma4](./models/Gemma4) ·
[GLM4V](./models/GLM4V) ·
[InternVL2](./models/InternVL2) ·
[InternVL3](./models/InternVL3) ·
[Janus-Pro](./models/Janus-Pro) ·
[Llama3_2-Vision](./models/Llama3_2-Vision) ·
[LocateAnything](./models/LocateAnything) ·
[MiniCPM-V-2_6](./models/MiniCPM-V-2_6) ·
[MiniCPMV4](./models/MiniCPMV4) ·
[MiniCPMV4_6](./models/MiniCPMV4_6) ·
[Molmo](./models/Molmo) ·
[NVILA](./models/NVILA) ·
[Qwen2_5_Omni](./models/Qwen2_5_Omni) ·
[Qwen2_5_VL](./models/Qwen2_5_VL) ·
[Qwen2_Audio](./models/Qwen2_Audio) ·
[Qwen2_VL](./models/Qwen2_VL) ·
[Qwen3_5](./models/Qwen3_5) ·
[Qwen3_ASR](./models/Qwen3_ASR) ·
[Qwen3_VL](./models/Qwen3_VL) ·
[VILA1_5](./models/VILA1_5)

See each subdirectory for complete source code and conversion details.

---

## 🧩 LLM Compilation Flow

Using `Qwen2.5-VL` as an example:

### 1. Download Weights

> Prefer **AWQ** or **GPTQ** quantized versions for better accuracy.

```bash
git lfs install
git clone git@hf.co:Qwen/Qwen2.5-VL-3B-Instruct-AWQ
```

### 2. Set Up TPU-MLIR

Refer to [TPU-MLIR](https://github.com/sophgo/tpu-mlir)

### 3. One-click Compilation to bmodel

```bash
llm_convert.py \
    -m /workspace/Qwen2.5-VL-3B-Instruct-AWQ \
    -s 2048 --max_input_length 1024 \
    -c bm1684x \
    --max_pixels 672,896 \
    -o qwen2.5vl_3b
```

#### Key `llm_convert.py` Arguments

| Argument | Short | Required | Description |
| :--- | :---: | :---: | :--- |
| `--model_path`         | `-m` | ✅ | Path to model weights |
| `--seq_length`         | `-s` | ✅ | Maximum sequence length |
| `--max_input_length`   |  —   |    | Maximum single-input length; defaults to `seq_length` |
| `--quantize`           | `-q` |    | Quantization type: `w4bf16` / `w4f16` / `bf16` / `f16` … |
| `--chip`               | `-c` | ✅ | Target platform: `bm1684x` / `bm1688` / `cv186x` |
| `--q_group_size`       | `-g` |    | Quantization group size, default `64` |
| `--max_pixels`         |  —   |    | VLM only, max pixels, e.g. `672,896` or `602112` |
| `--do_sample`          |  —   |    | Include sampling model in output; off by default |
| `--out_dir`            | `-o` | ✅ | Output directory |

> 💡 Choosing quantization: if the model is already quantized, you do NOT need to specify `quantize`; unquantized models require it.
>
> For more advanced options, see [Advanced Features](#-advanced-features).

Once finished, the output directory will contain the corresponding **bmodel** and **config** directory, ready to load for inference.

---

## ⚙️ Advanced Features

<table>
<thead>
<tr><th>Capability</th><th>Description</th><th>How to Enable</th><th>Examples</th></tr>
</thead>
<tbody>

<tr>
<td><b>Dynamic Compilation</b></td>
<td>Runs inference based on actual input length, reducing latency for short inputs; also recommended for variable-size multimodal images</td>
<td><code>--dynamic</code></td>
<td>
<a href="./models/Qwen3">Qwen3</a> · <a href="./models/Qwen2_5_VL">Qwen2.5-VL</a> · <a href="./models/MiniCPM4">MiniCPM4</a> · <a href="./models/InternVL3">InternVL3</a> · <a href="./models/Qwen3_VL">Qwen3-VL</a>
</td>
</tr>

<tr>
<td><b>Prefill with KV Cache</b></td>
<td>Keeps historical context as KV Cache, significantly reducing multi-turn conversation latency</td>
<td><code>--use_block_with_kv</code><br/><code>--max_input_length</code><br/><code>--max_prefill_kv_length</code></td>
<td>
<a href="./models/Qwen3_VL">Qwen3-VL</a> · <a href="./models/Qwen2_5_VL">Qwen2.5-VL</a> · <a href="./models/Qwen3">Qwen3</a> · <a href="./models/InternVL3">InternVL3</a>
</td>
</tr>

<tr>
<td><b>Multi-chip Parallelism</b></td>
<td>Parallel inference across multiple TPUs, enabling larger models and higher throughput</td>
<td><code>--num_device N</code></td>
<td>
<a href="./models/Qwen2_5/python_demo_parallel">Qwen2.5 / 2-8 chips</a>
</td>
</tr>

<tr>
<td><b>Random Sampling</b></td>
<td>Sampling with <code>generation.json</code> configuration (greedy by default)</td>
<td><code>--do_sample</code></td>
<td>
<a href="./models/Qwen3">Qwen3</a> · <a href="./models/InternVL3">InternVL3</a> · <a href="./models/MiniCPM4">MiniCPM4</a>
</td>
</tr>

<tr>
<td><b>Multi-task Reuse</b></td>
<td>Load the same model multiple times for multiple tasks; weights are loaded only once per chip</td>
<td>—</td>
<td>
<a href="./models/Qwen2_5_VL/cpp_demo_multiuser/">Qwen2.5-VL multiuser</a>
</td>
</tr>

<tr>
<td><b>Shared Prefill Reuse</b></td>
<td>Prefill a long prompt only once; subsequent conversations share its KV Cache</td>
<td><code>--share_prompt</code><br/><code>--max_prefill_kv_length</code></td>
<td>
<a href="./models/Qwen2_5/python_demo_share_prompt">Qwen2.5</a> · <a href="./models/Qwen3/python_demo_share_prompt">Qwen3</a>
</td>
</tr>

<tr>
<td><b>Model Encryption</b></td>
<td>Encrypt bmodel with a third-party library; call the decryption API at inference time</td>
<td>—</td>
<td>
<a href="./models/Qwen/share_cache_demo">Qwen</a> · <a href="./models/Qwen1_5/share_cache_demo">Qwen1.5</a>
</td>
</tr>

</tbody>
</table>

---

## 🎯 Accuracy Optimization Tips

1. **Prefer AWQ / GPTQ quantized models** when converting to bmodel — they incur the least accuracy loss.
2. If only floating-point weights are available, first apply W4A16 quantization with [AutoAWQ](https://huggingface.co/docs/transformers/main/en/quantization/awq#awq) or [AutoGPTQ](https://huggingface.co/docs/transformers/main/en/quantization/gptq), then compile to bmodel.

---

## ❓ FAQ

Please refer to the **[LLM-TPU FAQ](./docs/FAQ.md)**.

---

## 🔗 Resources

- 📘 [TPU-MLIR](https://github.com/sophgo/tpu-mlir) — Main compiler repository
- 📗 [TPU-MLIR Quick Start Guide](https://doc.sophgo.com/sdk-docs/v23.09.01-lts-sp4/docs_latest_release/docs/tpu-mlir/quick_start/html/index.html)
- 🎬 [TPU-MLIR Paper / Full Project Walkthrough (Bilibili)](https://www.bilibili.com/video/BV1My4y1o73Q)
- ✍️ [ChatGLM2 Pipeline Analysis & TPU-MLIR Deployment (Zhihu)](https://zhuanlan.zhihu.com/p/641975976)
- 🌐 [SOPHGO Official Website](https://www.sophgo.com/)

---

## 🤝 Contributing & Feedback

Issues and suggestions are welcome via [GitHub Issues](https://github.com/sophgo/LLM-TPU/issues), and Pull Requests are appreciated to help grow the ecosystem.
If you are interested in SOPHGO chips or business cooperation, feel free to reach out through the [SOPHGO website](https://www.sophgo.com/).

## 📄 License

This project is open-sourced under the [Apache 2.0](./LICENSE) license. See [`third-party-licenses/`](./third-party-licenses) for third-party component licenses.

<div align="center">

**⭐ If this project helps you, please give it a Star! ⭐**

</div>
