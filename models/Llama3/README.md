# Llama3

This project deploys the large model [Llama3](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) on BM1684X/BM1688. The model is converted into a bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed to a PCIE or SoC environment using C++ code.


This document covers how to compile the bmodel and how to run the bmodel in the BM1684X/BM1688 environment. The LLM compilation step can be skipped by downloading directly from the following links:

``` shell
# 1684x 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama-3.2-3b-instruct_w4f16_seq512_bm1684x_1dev_20250526_160605.bmodel
# 1688 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/llama-3.2-3b-instruct_w4f16_seq512_bm1688_2core_20250526_161500.bmodel
```

## Compiling the LLM

This section describes how to compile the LLM into a bmodel.

#### 1. Download `Llama3-3B` from HuggingFace

(The model is quite large and will take a long time)

``` shell
# Download the model
git lfs install
git clone git@hf.co:meta-llama/Llama-3.2-3B-Instruct
```

#### 2. Download docker and start the container

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
The following sections assume the environment is in the `/workspace` directory inside docker.

#### 2. Download the `TPU-MLIR` code and compile it

(You can also directly download and extract the precompiled release package)

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  # activate environment variables
./build.sh # compile mlir
```

#### 3. Compile the model to generate the bmodel

``` shell
# If you are prompted about a transformers version issue, run pip3 install transformers -U
# To compile for bm1688, simply use -c bm1688
llm_convert.py -m /workspace/Llama-3.2-3B-Instruct -s 512 -q w4f16 -g 128 -c bm1684x -o llama3.2_3b
```
After compilation completes, `llama-3.2-3bxxxx.bmodel` and `config` are generated in the specified directory `llama3.2_3b`

## Compiling and running the program

Please copy the program to the PCIE or SoC environment before compiling. Then copy `llama-3.2-3bxxxx.bmodel` and `config` over as well.

#### python demo

Compile the library to generate the `chat.cpython*.so` file, then copy it to the directory containing `pipeline.py`

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

* python demo

``` shell
python3 pipeline.py -m llama3_xxx.bmodel -c config 
```
`model` is the actual path where the model is stored; `config` is the path of the configuration file
