# MiniCPM4

This project implements the deployment of the large model [MiniCPM4](https://huggingface.co/openbmb/MiniCPM4-0.5B-QAT-Int4-GPTQ-format) on BM1684X/BM1688. The model is converted into a bmodel via the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed to a PCIE environment or a SoC environment using C++ code.


This document covers how to compile the bmodel and how to run the bmodel in the BM1684X/BM1688 environment. The LLM compilation step can be skipped; download directly using the following links:

``` shell
# minicpm4-8b 1684x 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm4-8b_w4bf16_seq512_bm1684x_1dev_20250613_175044.bmodel
# minicpm4-8b 1684x 8k, dynamic model
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm4-8b_w4bf16_seq8192_bm1684x_1dev_20250613_182940.bmodel

# minicpm4-0.5b bm1688 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/minicpm4-0.5b-gptq_w4bf16_seq512_bm1688_2core_20250616_122001.bmodel
# minicpm4-0.5b cv186x 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/minicpm4-0.5b-gptq_w4bf16_seq512_cv186x_1core_20250616_122126.bmodel

```

## Compile the LLM Model

This section describes how to compile an LLM into a bmodel.

#### 1. Download MiniCPM4 from HuggingFace

(Relatively large; will take a long time)

``` shell
# Download the model
git lfs install
git clone git@hf.co:openbmb/MiniCPM4-0.5B-QAT-Int4-GPTQ-format
# For 8B, use the following:
git clone git@hf.co:openbmb/MiniCPM4-8B
```

#### 2. Download docker and start the container

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
The following assumes that the environment is in the `/workspace` directory of the docker container.

#### 2. Download the `TPU-MLIR` code and compile it

(You can also directly download and extract the pre-compiled release package)

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  # activate the environment variables
./build.sh # compile mlir
```

#### 3. Compile the model to generate the bmodel

``` shell
# If you are prompted about a transformers version issue, run pip3 install transformers -U
llm_convert.py -m /workspace/MiniCPM4-0.5B-QAT-Int4-GPTQ-format -s 512 --quantize w4bf16 -c bm1684x --out_dir minicpm4_0.5b
```
After compilation, `minicpm4-xxx.bmodel` and `config` are generated in the specified directory `minicpm4_0.5b`

In addition, if the specified seqlen is relatively long, such as 8K, you can specify `--dynamic` compilation; the first-token latency will then vary with the actual length, as follows:
``` shell
# If you are prompted about a transformers version issue, run pip3 install transformers -U
llm_convert.py -m /workspace/MiniCPM4-0.5B-QAT-Int4-GPTQ-format -s 8192 --quantize w4bf16 -c bm1684x --dynamic --out_dir minicpm4_0.5b
```

## Compile and Run the Program

Please copy the program to the PCIE environment or SoC environment before compiling. Then copy `minicpm4-xxx.bmodel` and `config` over.

#### python demo

Compile the library files to generate the `chat.cpython*.so` file, and copy this file to the directory containing `pipeline.py`

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

* python demo

``` shell
python3 pipeline.py -m minicpm4_xxx.bmodel -c config 
```
model is the actual model storage path; config is the configuration file path
