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
| 🔥 **2026.07.16** | **Falcon-Perception** now supports BM1684X — Python demo for referring segmentation (box + mask) → [Details](./models/Falcon-Perception/) |
| 🔥 **2026.07.09** | **LocateAnything-3B** now supports BM1684X / BM1688 — Python demo for visual grounding (box / point) → [Details](./models/LocateAnything/) |
| **2026.06.30** | **MiniCPM-V-4.6** now supports BM1684X / BM1688 — Python demo with image & video support → [Details](./models/MiniCPMV4_6/) |
| **2026.05.21** | **Gemma4** now supports BM1684X / BM1688 — Python demo with image / video / audio support → [Details](./models/Gemma4/) |
| **2026.04.15** | **Qwen3.5** now supports BM1684X / BM1688 — Python & C++ demos with image & video support → [Details](./models/Qwen3_5/) |
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
[ChatGLM3](./models/ChatGLM3) ·
[Llama3](./models/Llama3) ·
[MiniCPM4](./models/MiniCPM4) ·
[Phi-3](./models/Phi-3) ·
[Qwen2_5](./models/Qwen2_5) ·
[Qwen3](./models/Qwen3)

**Multimodal (Vision / Video / Audio)**:
[Falcon-Perception](./models/Falcon-Perception) ·
[Gemma3](./models/Gemma3) ·
[Gemma4](./models/Gemma4) ·
[GLM4V](./models/GLM4V) ·
[InternVL3](./models/InternVL3) ·
[Janus-Pro](./models/Janus-Pro) ·
[Llama3_2-Vision](./models/Llama3_2-Vision) ·
[LocateAnything](./models/LocateAnything) ·
[MiniCPMV4](./models/MiniCPMV4) ·
[MiniCPMV4_6](./models/MiniCPMV4_6) ·
[NVILA](./models/NVILA) ·
[Qwen2_5_Omni](./models/Qwen2_5_Omni) ·
[Qwen2_5_VL](./models/Qwen2_5_VL) ·
[Qwen2_VL](./models/Qwen2_VL) ·
[Qwen3_5](./models/Qwen3_5) ·
[Qwen3_ASR](./models/Qwen3_ASR) ·
[Qwen3_VL](./models/Qwen3_VL)

#### Legacy

Older demos built with the pre-`llm_convert.py` compile flow (ONNX export + `model_transform.py` / `model_deploy.py`) are kept under [`models/legacy/`](./models/legacy) for reference and are no longer actively maintained:

**LLM**: [Baichuan2](./models/legacy/Baichuan2) · [ChatGLM2](./models/legacy/ChatGLM2) · [CodeFuse](./models/legacy/CodeFuse) · [DeepSeek-V2](./models/legacy/DeepSeek-V2) · [GLM4](./models/legacy/GLM4) · [Llama2](./models/legacy/Llama2) · [LWM](./models/legacy/LWM) · [Megrez](./models/legacy/Megrez) · [MiniCPM3](./models/legacy/MiniCPM3) · [Mistral](./models/legacy/Mistral) · [Qwen](./models/legacy/Qwen) · [Qwen1_5](./models/legacy/Qwen1_5) · [Qwen2](./models/legacy/Qwen2) · [RWKV6](./models/legacy/RWKV6) · [RWKV7](./models/legacy/RWKV7) · [WizardCoder](./models/legacy/WizardCoder) · [Yi](./models/legacy/Yi) · [Yi34B](./models/legacy/Yi34B)

**Multimodal**: [DriveMM](./models/legacy/DriveMM) · [InternVL2](./models/legacy/InternVL2) · [MiniCPM-V-2_6](./models/legacy/MiniCPM-V-2_6) · [Molmo](./models/legacy/Molmo) · [Qwen2_Audio](./models/legacy/Qwen2_Audio) · [VILA1_5](./models/legacy/VILA1_5)

See each subdirectory for complete source code and conversion details.

---

## 🧩 LLM Compilation Flow

Using `Qwen3.5-2B` as an example:

### 1. Download Weights

> Prefer **AWQ** / **GPTQ** / **AutoRound** quantized versions for better accuracy.

```bash
git lfs install
git clone https://huggingface.co/Intel/Qwen3.5-2B-int4-AutoRound
```

### 2. Set Up TPU-MLIR

Refer to [TPU-MLIR](https://github.com/sophgo/tpu-mlir)

### 3. One-click Compilation to bmodel

```bash
llm_convert.py \
    -m /workspace/Qwen3.5-2B-int4-AutoRound \
    -s 2048 --max_input_length 1024 \
    -c bm1684x \
    -o qwen3.5_2b
```

#### Two compile modes: without history vs. with history

All LLM compile scenarios fall into two categories, controlled by `--use_history_kv`:

**Mode 1 — Without history.** Typical command:

```bash
llm_convert.py -m Qwen3.5-2B-int4-AutoRound -c bm1688 -s 2048 --max_input_length 1024 --out_dir qwen3_5_bm1688
```

- Compiles two kinds of nets: `block_*` (prefill) and `block_cache_*` (decode).
- `-s` sets the maximum total length; `--max_input_length` sets the maximum single-input length.
- Recommended for single-turn conversations with short contexts (e.g. within 4K).

**Mode 2 — With history.** Typical command:

```bash
llm_convert.py -m Qwen3.5-2B-int4-AutoRound -c bm1688 -s 8192 --use_history_kv --chunk_length 1024 --out_dir qwen3_5_bm1688
```

- Compiles three kinds of nets: `block_*` (prefill), `block_kv_*` (prefill with history KV), and `block_cache_*` (decode).
- `-s` sets the maximum total length; `--chunk_length` sets the segment length used for chunked inference. For example, with `--chunk_length 1024` and a 7K-token input, prefill runs in 7 chunk passes: the first through `block_`, the remaining 6 through `block_kv_`. Decode is also segmented by KV-cache length, so performance at 1K / 2K / 4K / 8K varies with the context length.
- Recommended whenever multi-turn history is needed, contexts are long (e.g. 8K), or you are unsure — it is more flexible while retaining good performance.

#### Key `llm_convert.py` Arguments

| Argument | Short | Required | Description |
| :--- | :---: | :---: | :--- |
| `--model_path`         | `-m` | ✅ | Path to model weights |
| `--seq_length`         | `-s` | ✅ | Maximum total sequence length (KV cache capacity) |
| `--max_input_length`   |  —   |    | Maximum single-input length; defaults to `seq_length`. Do **not** set with `--use_history_kv` (there it is derived from `--chunk_length`) |
| `--use_history_kv`     |  —   |    | Compile with history-KV support (multi-turn); see the two compile modes above |
| `--chunk_length`       |  —   |    | Segment length for chunked prefill/decode; with `--use_history_kv` it defaults to `seq_length // 4` |
| `--chip`               | `-c` |    | Target platform: `bm1684x` (default) / `bm1688` / `cv186x` |
| `--dynamic`            |  —   |    | Dynamic compilation — recommended to always add |
| `--do_sample`          |  —   |    | Enable random sampling; off by default (greedy) |
| `--out_dir`            | `-o` |    | Output directory; defaults to `./<model>_<chip>_<quantize>` |

> 💡 Choosing quantization: if the model is already quantized, you do NOT need to specify `quantize`; unquantized models require it.
>
> For advanced options (`--quantize`, `--q_group_size`, `--max_pixels`, `--embedding_disk`, `--lora_max_rank`), see [Advanced Compile Options](#advanced-compile-options); for more capabilities, see [Advanced Features](#-advanced-features).

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
<td><code>--use_history_kv</code><br/><code>--chunk_length</code></td>
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
<td><code>--use_history_kv</code></td>
<td>
<a href="./models/Qwen2_5/python_demo_share_prompt">Qwen2.5</a> · <a href="./models/Qwen3/python_demo_share_prompt">Qwen3</a> · <a href="./models/Qwen3_5/cpp_demo_share_prompt">Qwen3.5</a>
</td>
</tr>

<tr>
<td><b>Model Encryption</b></td>
<td>Encrypt bmodel with a third-party library; call the decryption API at inference time</td>
<td>—</td>
<td>
<a href="./models/legacy/Qwen/share_cache_demo">Qwen</a> · <a href="./models/legacy/Qwen1_5/share_cache_demo">Qwen1.5</a>
</td>
</tr>

</tbody>
</table>

### Advanced Compile Options

Less commonly used `llm_convert.py` arguments:

| Argument | Short | Description |
| :--- | :---: | :--- |
| `--quantize`       | `-q` | Quantization type: `w4bf16` / `w4f16` / `bf16` / `f16` … |
| `--q_group_size`   | `-g` | Quantization group size, default `64` |
| `--max_pixels`     |  —   | VLM only, max image pixels, e.g. `672,896` or `602112`; recommended to leave unset and use the built-in default |
| `--embedding_disk` |  —   | Store the word embedding in a `.bin` file and run it on CPU |
| `--lora_max_rank`  |  —   | Maximum LoRA rank; setting it compiles the LoRA variant (Qwen3.5 LoRA support is not tuned yet) |

---

## Using the Demo

The interactive demos support a few convenience inputs:

- **Slash commands** — enter `/exit` (or `/q`, `/quit`) to quit the demo, and `/clear` (or `/new`) to start a new chat session.
- **`@<path>` attachments** — include `@<path>` in your question to attach a file:
  - Images (and videos, where the model supports them), e.g. `what is the image about? @./test.jpg`
  - Text files (`.txt` / `.md`), e.g. `what is it talking about? @./story.txt`

---

## 🎯 Accuracy Optimization Tips

1. **Prefer AWQ / GPTQ / AutoRound quantized models** when converting to bmodel — they incur the least accuracy loss.
2. If only floating-point weights are available, first apply W4A16 quantization with [AutoAWQ](https://huggingface.co/docs/transformers/main/en/quantization/awq#awq) or [AutoGPTQ](https://huggingface.co/docs/transformers/main/en/quantization/gptq), then compile to bmodel.

---

## ❓ FAQ

Please refer to the **[LLM-TPU FAQ](./docs/FAQ.md)**.

---

## 🔗 Resources

- 📄 [An MLIR-Based Compilation Method for Large Language Models](https://arxiv.org/abs/2607.15865) — Paper describing the TPU-MLIR LLM compilation flow
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
