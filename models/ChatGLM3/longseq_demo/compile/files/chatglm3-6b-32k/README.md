---
language:
- zh
- en
tags:
- glm
- chatglm
- thudm
---
# ChatGLM3-6B-32K
<p align="center">
  💻 <a href="https://github.com/THUDM/ChatGLM" target="_blank">Github Repo</a> • 🐦 <a href="https://twitter.com/thukeg" target="_blank">Twitter</a> • 📃 <a href="https://arxiv.org/abs/2103.10360" target="_blank">[GLM@ACL 22]</a> <a href="https://github.com/THUDM/GLM" target="_blank">[GitHub]</a> • 📃 <a href="https://arxiv.org/abs/2210.02414" target="_blank">[GLM-130B@ICLR 23]</a> <a href="https://github.com/THUDM/GLM-130B" target="_blank">[GitHub]</a> <br>
</p>

<p align="center">
    👋 Join our <a href="https://join.slack.com/t/chatglm/shared_invite/zt-25ti5uohv-A_hs~am_D3Q8XPZMpj7wwQ" target="_blank">Slack</a> and <a href="https://github.com/THUDM/ChatGLM/blob/main/resources/WECHAT.md" target="_blank">WeChat</a>
</p>
<p align="center">
📍Experience the larger-scale ChatGLM model at <a href="https://www.chatglm.cn">chatglm.cn</a>
</p>

## Introduction
ChatGLM3-6B-32K further strengthens the long-text understanding capability on top of [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b), enabling better handling of contexts up to 32K in length. Specifically, we updated the positional encoding and designed a more targeted long-text training method, using a 32K context length during the dialogue training stage. In practice, if your context length is generally **within 8K**, we recommend using [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b); if you need to handle context lengths **exceeding 8K**, we recommend using ChatGLM3-6B-32K.


ChatGLM3-6B is the latest generation of the open-source ChatGLM series. While retaining many excellent features of the previous two generations, such as smooth dialogue and a low deployment barrier, ChatGLM3-6B introduces the following features:

1. **More powerful base model:** The base model of ChatGLM3-6B, ChatGLM3-6B-Base, adopts more diverse training data, more sufficient training steps, and a more reasonable training strategy. Evaluations on datasets covering semantics, mathematics, reasoning, code, knowledge, and other dimensions show that ChatGLM3-6B-Base delivers the strongest performance among pre-trained models under 10B parameters.
2. **More complete feature support:** ChatGLM3-6B adopts a newly designed [Prompt format](PROMPT.md). In addition to normal multi-turn conversation, it natively supports [tool calling](tool_using/README.md) (Function Call), code execution (Code Interpreter), and complex scenarios such as Agent tasks.
3. **More comprehensive open-source lineup:** In addition to the dialogue model ChatGLM3-6B, the base model ChatGLM-6B-Base and the long-text dialogue model ChatGLM3-6B-32K have also been open-sourced. All of the above weights are **fully open** for academic research, and **free commercial use is also allowed** after registering by filling out the [questionnaire](https://open.bigmodel.cn/mla/form).


## Software Dependencies

```shell
pip install protobuf transformers==4.30.2 cpm_kernels torch>=2.0 gradio mdtex2html sentencepiece accelerate
```

## Model Download

Download via the ModelScope API
```shell
pip install modelscope
```

```python
from modelscope import snapshot_download
model_dir = snapshot_download("ZhipuAI/chatglm3-6b-32k", revision = "master")
```

Download via git
```shell
git lfs install
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b-32k.git
```




## Code Usage

You can use the following code to call the ChatGLM3-6B model to generate dialogue:

```python
from modelscope import AutoTokenizer, AutoModel, snapshot_download
model_dir = snapshot_download("ZhipuAI/chatglm3-6b-32k", revision = "master")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
```

For more usage instructions, including how to run the CLI and web demos, and how to use model quantization to save GPU memory, please refer to our [Github Repo](https://github.com/THUDM/ChatGLM).

For more instructions, including how to run CLI and web demos, and model quantization, please refer to our [Github Repo](https://github.com/THUDM/ChatGLM).


## License

The code in this repository is open-sourced under the [Apache-2.0](LICENSE) license, while the use of the ChatGLM3-6B model weights must comply with the [Model License](MODEL_LICENSE).

## Citation

If you find our work helpful, please consider citing the following papers.

```
@article{zeng2022glm,
  title={Glm-130b: An open bilingual pre-trained model},
  author={Zeng, Aohan and Liu, Xiao and Du, Zhengxiao and Wang, Zihan and Lai, Hanyu and Ding, Ming and Yang, Zhuoyi and Xu, Yifan and Zheng, Wendi and Xia, Xiao and others},
  journal={arXiv preprint arXiv:2210.02414},
  year={2022}
}
```
```
@inproceedings{du2022glm,
  title={GLM: General Language Model Pretraining with Autoregressive Blank Infilling},
  author={Du, Zhengxiao and Qian, Yujie and Liu, Xiao and Ding, Ming and Qiu, Jiezhong and Yang, Zhilin and Tang, Jie},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={320--335},
  year={2022}
}
```
