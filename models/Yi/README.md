![](./assets/tpumlir.png)

# Yi

This project implements the deployment of the large language model [Yi-6B-Chat](https://huggingface.co/01-ai/Yi-6B-Chat) on BM1684X. The model is converted into a bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed with C++ code to the BM1684X PCIE environment or SoC environment.


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

sudo apt-get update
sudo apt-get install pybind11-dev
```
```

### 4. Download the pytorch.bin model

``` shell
cd LLM-TPU/models/Yi/
git lfs install
git clone https://huggingface.co/01-ai/Yi-6B-Chat
cp compile/files/Yi-6B-Chat/config.json Yi-6B-Chat
cp compile/files/Yi-6B-Chat/modeling_llama.py /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
export PYTHONPATH=$PWD/Yi-6B-Chat:$PYTHONPATH

cd compile
python3 export_onnx.py --model_path ../Yi-6B-Chat
```

This project is quite large and will take a long time.
Before exporting ONNX, make sure the files in `files/Yi-6B-Chat` have replaced the corresponding files under the `transformers` package actually used at runtime. (The default sequence length is 512.)

## 5. Compile the model

Note that you should be in the workspace directory of the Docker environment at this point.

TPU-MLIR currently supports FP16, INT8 and INT4 quantization for `Yi-6B-Chat`, and supports multi-chip distributed inference. By default, INT8 quantization and single-chip inference are performed, finally generating the `yi-6b_int4_1dev.bmodel` file. (Please make sure you have already performed [the mlir compilation and environment activation](#2-download-the-tpu-mlir-code-and-compile-it) first.)

```shell
cd LLM-TPU/models/Yi-6B-Chat/compile
./compile.sh --name yi-6b --mode int4 --addr_mode io_alone # int4 (defaulted)
```

If you want to perform 2-chip inference, run the following command, which finally generates the `yi-6b_int4_1dev.bmodel` file; the same applies to 4-chip and 8-chip inference:

```shell
./compile.sh --num_device 2 --name yi-6b --mode int4 --addr_mode io_alone
```

## 6. Using the models provided by SOPHGO
You can also use the models already compiled by SOPHGO for subsequent inference, as follows:
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/yi-6b_int4_1dev.bmodel
```

## 7. Compile the program (Python)

Run the following compilation (note that for the SoC version, you need to copy the demo directory to the SoC environment to compile):

```shell
cd /workspace/LLM-TPU/models/Yi/python_demo
mkdir build
cd build
cmake ..
make
cp chat.cpython-310-x86_64-linux-gnu.so ..
cd ..
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
