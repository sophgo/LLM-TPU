![](./assets/tpumlir.png)

# Yi

This project implements the deployment of the large language model [Yi-34B-Chat](https://huggingface.co/01-ai/Yi-34B-Chat) on BM1684X. The model is converted into a bmodel via the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed to the multi-chip PCIE environment of BM1684X using C++ code.


## Development Environment Setup

### 1. Download docker and start the container

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```
The following assumes that the environment is in the `/workspace` directory of the docker container.

### 2. Download the `TPU-MLIR` code and compile it

(You can also directly download and extract the pre-compiled release package)

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  # activate the environment variables
./build.sh # compile mlir
```

### 3. Update third-party libraries

Download this project:
``` shell
git clone git@github.com:sophgo/LLM-TPU.git
```
Third-party library environment requirements
``` shell
pip3 install transformers==4.39.1
pip3 install torch==2.0.1

sudo apt-get update
sudo apt-get install pybind11-dev
```
```

### 4. Download the pytorch.bin model

``` shell
cd LLM-TPU/models/Yi34B/
git lfs install
git clone https://huggingface.co/01-ai/Yi-34B-Chat
cp compile/files/Yi-34B-Chat/modeling_llama.py /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
export PYTHONPATH=$PWD/Yi-34B-Chat:$PYTHONPATH

cd compile
python3 export_onnx.py --model_path ../Yi-34B-Chat
```

This project is relatively large and will take a long time.
Before exporting onnx, make sure the files in `files/Yi-34B-Chat` have replaced the corresponding files under the `transformers` package actually used at runtime. (The default sequence length is 512)

## 5. Compile the Model

Make sure you are in the workspace directory of the Docker environment at this point.

Currently TPU-MLIR supports INT8 and INT4 quantization of `Yi-34B-Chat`, and only supports multi-chip distributed inference. By default, INT4 quantization and dual-chip inference are performed, ultimately generating the `yi-34b_int4_2dev.bmodel` file. (Please make sure you have already performed the [mlir compilation and environment activation](#2-download-the-tpu-mlir-code-and-compile-it) first).

To perform 2-chip inference, run the following command, which ultimately generates the `yi-34b_int4_2dev.bmodel` file. The same applies to 4-chip and 8-chip:

```shell
./compile.sh --num_device 2 --name yi-34b --mode int4
```

## 6. Use the Models Provided by Sophgo
*The following models have a maximum context length of 512
You can also use the models already compiled by Sophgo for subsequent inference, as follows:
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/yi-34b_int4_2dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/yi-34b_int4_4dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/yi-34b_int4_8dev.bmodel
```

## 7. Running python_demo

For detailed commands, please refer to python_demo/README.md

## 8. Running demo_parallel

For detailed commands, please refer to demo_parallel/README.md (Currently, due to an issue with the official sentencepiece/tokenizer.model, this C++ demo temporarily does not support normal inference)
