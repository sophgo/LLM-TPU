# Megrez-3B-Instruct

This project deploys the large language model [Megrez-3B-Instruct](https://huggingface.co/Infinigence/Megrez-3B-Instruct) on BM1684X. The model is converted to bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed to the PCIE environment or SoC environment of BM1684X using C++ code.

## Development Environment Setup

#### 1. Download `Megrez-3B-Instruct` from HuggingFace

``` shell
git lfs install
git clone https://huggingface.co/Infinigence/Megrez-3B-Instruct
```

Some modifications to the model source code are also required:
* Replace the `modeling_llama.py` file in transformers with the `modeling_llama.py` under `compile/files`

#### 2. Export to onnx model

If you are prompted that some components are missing during the process, just `pip3 install component`

``` shell
# export onnx
cd compile
python3 export_onnx.py --model_path your_model_path
```

## Compile the Model

This section describes how to compile the onnx model into bmodel. You can also skip the compilation step and directly download the compiled model:

``` shell
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/megrez_bm1684x_int4_seq512.bmodel
```

#### 1. Download docker and start the container

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```
The following assumes that all environments are in the `/workspace` directory of docker.

#### 2. Install `TPU-MLIR`

``` shell
pip3 install tpu-mlir
```

#### 3. Compile the model to generate bmodel

Compile the ONNX model to generate the model

For details, please refer to python_demo/README.md

## Build and Run the Program

Compile the library files to generate the `chat.cpython*.so` file, and copy it to the directory of the `pipeline.py` file

```
cd python_demo
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

Run the program as follows:

```
python3 pipeline.py --model_path megrez_bm1684x_int4_seq512.bmodel --tokenizer_path ../support/token_config --devid 0
```
model_path is the actual storage path of the model; tokenizer_path is the actual storage path of the tokenizer configuration
