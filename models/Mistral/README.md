![](./assets/tpumlir.png)

# Mistral

This project implements the deployment of the large language model [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) on BM1684X. The model is converted into a bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed with C++ code to the BM1684X PCIE environment or SoC environment.


## Development Environment Setup

### 1. Download docker and start the container

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```
The following assumes the environment is in the `/workspace` directory of the docker container.

### 2. Download the `TPU-MLIR` code and compile it

(You can also directly download and extract the prebuilt release package.)

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  # activate environment variables
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
```

### 4. Download the pytorch.bin model

``` shell
cd LLM-TPU/models/Mistral/
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
cp compile/files/Mistral-7B-Instruct-v0.2/config.json Mistral-7B-Instruct-v0.2/
cp compile/files/Mistral-7B-Instruct-v0.2/modeling_mistral.py /usr/local/lib/python3.10/dist-packages/transformers/models/mistral/modeling_mistral.py
export PYTHONPATH=$PWD/Mistral-7B-Instruct-v0.2:$PYTHONPATH

cd compile
python3 export_onnx.py --model_path ../Mistral-7B-Instruct-v0.2
```

This project is quite large and will take a long time.
Before exporting ONNX, make sure the files in `files/Mistral-7B-Instruct-v0.2` have replaced the corresponding files in `Mistral-7B-Instruct-v0.2`. (The default sequence length is 512.)

## 5. Compile the model

Note that you should be in the workspace directory of the Docker environment at this point.

TPU-MLIR currently supports FP16, INT8 and INT4 quantization for `Mistral-7B-Instruct-v0.2`, and supports multi-chip distributed inference. By default, INT8 quantization and single-chip inference are performed, finally generating the `mistral-7b_int4_1dev.bmodel` file. (Please make sure you have already performed [the mlir compilation and environment activation](#2-download-the-tpu-mlir-code-and-compile-it) first.)

```shell
cd LLM-TPU/models/Mistral-7B-Instruct-v0.2/compile
./compile.sh --name mistral-7b --mode int4 --addr_mode io_alone # int4 (defaulted)
```

If you want to perform 2-chip inference, run the following command, which finally generates the `mistral-7b_int4_2dev.bmodel` file; the same applies to 4-chip and 8-chip inference:

```shell
./compile.sh --num_device 2 --name mistral-7b --mode int4 --addr_mode io_alone
```

## 6. Using the models provided by SOPHGO
You can also use the models already compiled by SOPHGO for subsequent inference, as follows:
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/mistral-7b_int4_1dev.bmodel
```

## 7. Compile the program (Python)

Run the following compilation (note that for the SoC version, you need to copy the demo directory to the SoC environment to compile):

```shell
cd /workspace/LLM-TPU/models/Mistral/python_demo
mkdir build
cd build
cmake ..
make
cp chat.cpython-310-x86_64-linux-gnu.so ..
cd ..
```
If you are prompted that `pybind` files are missing, use the following commands:
```shell
sudo apt-get update
sudo apt-get install pybind11-dev
```

### a. Command-line interaction
- Single-chip inference: use the following command.
```shell
python3 pipeline.py --model your_bmodel_path --devid 0 # devid defaults to device 0 for inference
```
Set `your_bmodel_path` according to the actual bmodel path. Other parameters can be viewed with:
```shell
python3 pipeline.py --help
```
