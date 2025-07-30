---
license: other
license_name: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-Omni-7B-AWQ/blob/main/LICENSE
language:
- en
tags:
- multimodal
library_name: transformers
pipeline_tag: any-to-any
---

# Qwen2.5-Omni-7B-AWQ
<a href="https://chat.qwen.ai/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/%F0%9F%92%9C%EF%B8%8F%20Qwen%20Chat%20-536af5" style="display: inline-block; vertical-align: middle;"/>
</a>


## Overview 
### Introduction
Qwen2.5-Omni is an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner. 

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/qwen_omni.png" width="80%"/>
<p>

### Key Features

* **Omni and Novel Architecture**: We propose Thinker-Talker architecture, an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner. We propose a novel position embedding, named TMRoPE (Time-aligned Multimodal RoPE), to synchronize the timestamps of video inputs with audio.

* **Real-Time Voice and Video Chat**: Architecture designed for fully real-time interactions, supporting chunked input and immediate output.

* **Natural and Robust Speech Generation**: Surpassing many existing streaming and non-streaming alternatives, demonstrating superior robustness and naturalness in speech generation.

* **Strong Performance Across Modalities**: Exhibiting exceptional performance across all modalities when benchmarked against similarly sized single-modality models. Qwen2.5-Omni outperforms the similarly sized Qwen2-Audio in audio capabilities and achieves comparable performance to Qwen2.5-VL-7B.

* **Excellent End-to-End Speech Instruction Following**: Qwen2.5-Omni shows performance in end-to-end speech instruction following that rivals its effectiveness with text inputs, evidenced by benchmarks such as MMLU and GSM8K.

### Model Architecture

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/overview.png" width="80%"/>
<p>

## Quickstart

This model card introduces a series of enhancements designed to improve the Qwen2.5-Omni-7B's operability on devices with constrained GPU memory. Key optimizations include:

* Implemented 4-bit quantization of the Thinker's weights using AWQ, effectively reducing GPU VRAM usage.

* Enhanced the inference pipeline to load model weights on-demand for each module and offload them to CPU memory once inference is complete, preventing peak VRAM usage from becoming excessive.

* Converted the token2wav module to support streaming inference, thereby avoiding the pre-allocation of excessive GPU memory.

* Adjusted the ODE solver from a second-order (RK4) to a first-order (Euler) method to further decrease computational overhead.

These improvements aim to ensure efficient performance of Qwen2.5-Omni across a range of hardware configurations, particularly those with lower GPU memory availability (RTX3080, 4080, 5070, etc).

Below, we provide simple example to show how to use Qwen2.5-Omni-7B-AWQ with `autoawq` as follows:
```
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
pip install accelerate
pip install autoawq==0.2.9

git clone https://github.com/QwenLM/Qwen2.5-Omni.git

cd Qwen2.5-Omni/low-VRAM-mode/

CUDA_VISIBLE_DEVICES=0 python3 low_VRAM_demo_awq.py
```

We offer a toolkit to help you handle various types of audio and visual input more conveniently, as if you were using an API. This includes base64, URLs, and interleaved audio, images and videos. You can install it using the following command and make sure your system has `ffmpeg` installed:

```bash
# It's highly recommended to use `[decord]` feature for faster video loading.
pip install qwen-omni-utils[decord] -U
```

If you are not using Linux, you might not be able to install `decord` from PyPI. In that case, you can use `pip install qwen-omni-utils -U` which will fall back to using torchvision for video processing. However, you can still [install decord from source](https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source) to get decord used when loading video.


### Performance and GPU memory requirements

The following two tables present a performance comparison and GPU memory consumption between Qwen2.5-Omni-7B-AWQ and Qwen2.5-Omni-7B on specific evaluation benchmarks. The data demonstrates that the AWQ model maintains comparable performance while reducing GPU memory requirements by over 50%+, enabling a broader range of devices to run and experience the high-performance Qwen2.5-Omni-7B model. Notably, the AWQ variant exhibits slightly slower inference speeds compared to the native Qwen2.5-Omni-7B model due to quantization techniques and CPU offload mechanisms.

| Evaluation Set | Task | Metrics | Qwen2.5-Omni-7B | Qwen2.5-Omni-7B-AWQ |
|--------------|-----------| ------------- | ------------- | ------------------ |
| LibriSpeech test-other   | ASR                   | WER ⬇️      | 3.4   | 3.91  |
| WenetSpeech test-net     | ASR                   | WER ⬇️      | 5.9   | 6.31  |
| Seed-TTS test-hard       | TTS (Speaker: Chelsie)| WER ⬇️      | 8.7   | 8.88  |
| MMLU-Pro                 | Text -> Text          | Accuracy ⬆️ | 47.0  | 45.66 |
| OmniBench                | Speech -> Text        | Accuracy ⬆️ | 56.13 | 54.64 |
| VideoMME                 | Multimodality -> Text | Accuracy ⬆️ | 72.4  | 72.0  |

|Model | Precision | 15(s) Video | 30(s) Video | 60(s) Video |
|--------------|-----------| ------------- | ------------- | ------------------ |
| Qwen-Omni-7B | FP32      | 93.56 GB      | Not Recommend | Not Recommend      |
| Qwen-Omni-7B | BF16      | 31.11 GB      | 41.85 GB      | 60.19 GB           |
| Qwen-Omni-7B | AWQ       | 11.77 GB      | 17.84 GB      | 30.31 GB           |


## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)


```BibTeX

@article{Qwen2.5-Omni,
  title={Qwen2.5-Omni Technical Report},
  author={Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, Junyang Lin},
  journal={arXiv preprint arXiv:2503.20215},
  year={2025}
}
```

<br>

