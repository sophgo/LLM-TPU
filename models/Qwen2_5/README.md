# Qwen2.5

This project implements the deployment of the large model [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) on BM1684X/BM1688. The model is converted into a bmodel via the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed to a PCIE environment or a SoC environment using C++ code.


This document covers how to compile the bmodel and how to run the bmodel in the BM1684X/BM1688 environment. The LLM compilation step can be skipped; download directly using the following links:

### Qwen2.5 series

``` shell
# qwen2.5-1.5b 2k
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-1.5b_int4_seq2048_1dev.bmodel
# qwen2.5-3b 2k
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-3b-instruct-gptq-int4_w4bf16_seq2048_bm1684x_1dev_20250620_134431.bmodel
# qwen2.5-3b 4k
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-3b_int4_seq4096_1dev.bmodel
# qwen2.5-7b 2k
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-7b-instruct-awq_w4bf16_seq2048_bm1684x_1dev_20250616_191537.bmodel
# qwen2.5-14b 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-14b_int4_seq512_1dev.bmodel
```

### QwQ-32B

``` shell
# qwq-32b 2k 2dev
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwq-32b-awq_int4_seq2048_2dev.bmodel
# qwq-32b 2k 4dev
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwq-32b-awq_int4_seq2048_4dev.bmodel
# qwq-32b 16k 4dev
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwq-32b-awq_w4bf16_seq16384_bm1684x_4dev_20250430_163346.bmodel
```

### deepseek-r1-distill-qwen series

``` shell
# deepseek-r1-distill-qwen-1.5b
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-distill-qwen-1-5b.zip
unzip deepseek-r1-distill-qwen-1-5b.zip

# deepseek-r1-distill-qwen-7b
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-distill-qwen-7b.zip
unzip deepseek-r1-distill-qwen-7b.zip

# deepseek-r1-distill-qwen-14b
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-distill-qwen-14b-seq512.zip
unzip deepseek-r1-distill-qwen-14b-seq512.zip

# deepseek-r1-distill-qwen-32b multi-chip
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-distill-qwen-32b-seq2048-4dev.zip
unzip deepseek-r1-distill-qwen-32b-seq2048-4dev.zip

# deepseek-r1-distill-qwen-32b multi-chip
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-distill-qwen-32b-seq4096-4dev.zip
unzip deepseek-r1-distill-qwen-32b-seq4096-4dev.zip

```

## Compile the LLM Model

``` shell
llm_convert.py -m /workspace/Qwen2.5-3B-Instruct -s 512 --quantize w4bf16 -c bm1684x --out_dir qwen2.5_3b
```
After compilation, `qwen2.5-3b_xxx.bmodel` and `config` are generated in the specified directory `qwen2.5_3b`, where config contains the original configuration files, the tokenizer, and so on.

In addition, if the specified seqlen is relatively long, such as 8K, you can specify `--dynamic` compilation; the first-token latency will then vary with the actual length, as follows:
``` shell
llm_convert.py -m /workspace/Qwen2.5-3B-Instruct -s 8192 --quantize w4bf16 -c bm1684x --dynamic --out_dir qwen2.5_3b
```

## Compile and Run the Program

Please copy the program to the PCIE environment or SoC environment before compiling. Then copy the `bmodel` and `config` files over.

#### python demo

Compile the library files to generate the `chat.cpython*.so` file, and copy this file to the directory containing `pipeline.py`

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

* python demo

``` shell
python3 pipeline.py -m qwen_xxx.bmodel -c config 
```
model is the actual storage path of the bmodel; config is the configuration file generated when compiling the model. The demo includes the config for qwen2.5.
* If you switch to another model series, such as deepseek-r1-distill-qwen, you need to specify the new config

#### python demo parallel

This demo is for multi-chip models; the usage is the same as for a single chip
