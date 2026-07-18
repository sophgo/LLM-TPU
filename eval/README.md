# Eval

This project implements accuracy testing for VLMs on `BM1684X/BM1688` as well as on `CUDA`.


## Download the model and dataset

You can use the following links to download the source model that runs on `CUDA`, the bmodel that runs on `BM1684X`, and the dataset used for testing. You can also compile the bmodel yourself.
```bash
# Qwen3-VL-2B-Instruct-W4A16 source model, run in the CUDA environment
git clone https://huggingface.co/kaitchup/Qwen3-VL-2B-Instruct-W4A16

# bmodel compiled based on the source model, run in the BM1684X environment, max_pixel 768x768
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-vl-2b-instruct-w4a16_w4bf16_seq2048_bm1684x_1dev_20251211_213351.bmodel

# Dataset A-OKVQA for accuracy testing, containing 17k samples
git clone https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA
```


## Source model accuracy test

This section describes how to run the program on a `CUDA` device to test the source model accuracy.

First, you need a `python3.10` or higher environment.

Install the following dependencies:

```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Then run the accuracy test program:
```bash
python eval_qwen3vl.py --model_path Qwen3-VL-2B-Instruct-W4A16 --datasets A-OKVQA
```

## bmodel accuracy test

This section describes how to run the program on a `BM1684X` device to test the bmodel accuracy.

First, you need a `python3.10` or higher environment.

Install the following dependencies:
```bash
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Then you need to compile the `Qwen3_VL/python_demo` program and copy the `*cpython*` file to the `python_demo` directory:
```bash
cd ../models/Qwen3_VL/python_demo
mkdir build && cd build && cmake .. && make && cp *cpython* .. && cd ..
```

Finally, return to this project directory and run the accuracy test program:
```bash
cd ../../eval
export PYTHONPATH=$PYTHONPATH:$(pwd)/../models/Qwen3_VL/python_demo
python eval_qwen3vl.py --model_path {your_bmodel_path.bmodel} --datasets A-OKVQA
```


* Notes:
1. It is recommended that the source model and the bmodel use the same quantization version, e.g. AWQ/W4A16 quantization.
2. If you compile the bmodel yourself, make sure the `--max_pixels` parameter used when compiling the bmodel is consistent with the `--max_pixels` parameter used when testing the source model.
