# Eval

本工程实现了`BM1684X/BM1688`上以及`CUDA`上对VLM精度的测试。


## 下载模型和数据集

可以用以下链接下载`CUDA`上运行的源模型、`BM1684X`上运行的bmodel、测试用的数据集。也可以自行编译bmodel。
```bash
# Qwen3-VL-2B-Instruct-W4A16 源模型，在CUDA环境运行
git clone https://huggingface.co/kaitchup/Qwen3-VL-2B-Instruct-W4A16

# 基于源模型编译的bmodel，在BM1684X环境运行，max_pixel 768x768
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-vl-2b-instruct-w4a16_w4bf16_seq2048_bm1684x_1dev_20251211_213351.bmodel

# 测试精度用的数据集 A-OKVQA，包含17k样本
git clone https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA
```


## 源模型精度测试

此处介绍如何在`CUDA`设备上运行程序，测试源模型精度。

首先需要`python3.10`及以上的环境。

安装以下依赖：

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

然后运行精度测试程序：
```bash
python eval_qwen3vl.py --model_path Qwen3-VL-2B-Instruct-W4A16 --datasets A-OKVQA
```

## bmodel精度测试

此处介绍如何在`BM1684X`设备上运行程序，测试bmodel精度。

首先需要`python3.10`及以上的环境。

安装以下依赖：
```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

然后需要编译`Qwen3_VL/python_demo`程序，并将`*cpython*`文件拷贝到`python_demo`目录下：
```bash
cd ../models/Qwen3_VL/python_demo
mkdir build && cd build && cmake .. && make && cp *cpython* .. && cd ..
```

最后回到本工程目录，运行精度测试程序：
```bash
cd ../../eval
export PYTHONPATH=$PYTHONPATH:$(pwd)/../models/Qwen3_VL/python_demo
python eval_qwen3vl.py --model_path {your_bmodel_path.bmodel} --datasets A-OKVQA
```


* 注意事项：
1. 建议源模型和bmodel使用同样的量化版本，例如AWQ/W4A16量化。
2. 若自行编译bmodel，请确保编译bmodel时的`--max_pixels`参数和测试源模型时的参数`--max_pixels`一致。
