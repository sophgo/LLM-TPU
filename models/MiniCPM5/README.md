# MiniCPM5

This project implements the deployment of the large model [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) on BM1684X/CV84X6. The model is converted into a bmodel via the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed to a PCIE environment or a SoC environment using C++ code.


This document covers how to compile the bmodel and how to run the bmodel in the BM1684X/CV84X6 environment.

## Compile the LLM Model

This section describes how to compile an LLM into a bmodel.

#### 1. Download MiniCPM5-2B from HuggingFace

(Relatively large; will take a long time)

```shell
git lfs install
git clone git@hf.co:openbmb/MiniCPM5-2B
# make sure the safetensors are fully downloaded (git lfs pull), not LFS pointer stubs
```

#### 2. Download docker and start the container

```shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
The following assumes that the environment is in the `/workspace` directory of the docker container.

#### 3. Download the `TPU-MLIR` code and compile it

(You can also directly download and extract the pre-compiled release package)

```shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  # activate the environment variables
./build.sh # compile mlir
```

#### 4. Compile the model to generate the bmodel

```shell
# If you are prompted about a transformers version issue, run pip3 install transformers -U
llm_convert.py -m /workspace/MiniCPM5-2B -s 512 --quantize bf16 -c bm1684x --out_dir minicpm5_2b
```

After compilation, `minicpm5-xxx.bmodel` and `config` are generated in the specified directory `minicpm5_2b`.

In addition, if the specified seqlen is relatively long, such as 8K, you can specify `--dynamic` compilation; the first-token latency will then vary with the actual length, as follows:
```shell
llm_convert.py -m /workspace/MiniCPM5-2B -s 8192 --quantize bf16 -c bm1684x --dynamic --out_dir minicpm5_2b
```

## Compile and Run the Program

Please copy the program to the PCIE environment or SoC environment before compiling. Then copy `minicpm5-xxx.bmodel` and `config` over.

#### python demo

Compile the library files to generate the `chat.cpython*.so` file, and copy this file to the directory containing `pipeline.py`

```shell
cd python_demo
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

* python demo

```shell
python3 pipeline.py -m minicpm5_xxx.bmodel -c config
```
model is the actual model storage path; config is the configuration file path
