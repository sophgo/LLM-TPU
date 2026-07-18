# MiniCPMV4

This project deploys the multimodal large model [MiniCPMV4](https://www.modelscope.cn/models/OpenBMB/MiniCPM-V-4-AWQ) on BM1684X. The model is converted to bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed to the PCIE environment or SoC environment of BM1684X using C++ code.

## Development Environment Setup

#### 1. Download `MiniCPM-V-4-AWQ` from ModelScope

(Relatively large, will take a long time)

``` shell
git lfs install
git clone https://www.modelscope.cn/OpenBMB/MiniCPM-V-4-AWQ.git
```

## Compile the Model

This section describes how to compile the onnx model into bmodel. You can also skip the compilation step and directly download the compiled model:

``` shell
# 1684x, 980x980
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm-v-4-awq_w4bf16_seq2048_bm1684x_1dev_20250915_204204.bmodel
```

#### 1. Download docker and start the container

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```
The following assumes that all environments are in the `/workspace` directory of docker.

#### 2. Download the `TPU-MLIR` code and compile it

(You can also directly download and extract the precompiled release package)

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  # activate environment variables
./build.sh # compile mlir
```

#### 3. Compile the model to generate bmodel

``` shell
# If prompted about transformers version issues, run pip3 install transformers --upgrade
llm_convert.py -m /workspace/MiniCPM-V-4-AWQ -s 2048 --quantize w4bf16 -c bm1684x --out_dir minicpmv --max_pixels 980,980
```
After compilation, `minicpm-v-4-xxx.bmodel` and `config` are generated in the specified directory `minicpmv`

## Build and Run the Program

Compile the library files to generate the `chat.cpython*.so` file, and copy it to the directory of the `pipeline.py` file;
also copy the bmodel file and config directory there

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

* python demo

``` shell
python3 pipeline.py -m minicpm-v-4-xxxx.bmodel -c ../config
```
