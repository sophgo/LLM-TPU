# Qwen1.5

## Table of Contents
  - [1. Introduction](#1-introduction)
  - [2. Features](#2-features)
  - [3. Environment Setup](#3-environment-setup)
  - [4. Model Compilation](#4-model-compilation)
  - [5. Model Inference](#5-model-inference)

## 1. Introduction
Qwen1.5 is the second generation of Qwen. It is an open-source Chinese-English bilingual chat model; for its features, please visit the source repo: https://huggingface.co/Qwen. This demo ports Qwen so that it can run inference tests on SOPHON BM1684X.

This demo supports SDK V23.07.01 (libsophon_0.4.9) and above. It can run on x86 hosts equipped with a 1684X accelerator card (SC7 series), and also on 1684X SoC devices (such as SE7, SM7, Airbox, etc.). Running on SoC requires additional environment configuration; please complete the environment deployment by referring to [Environment Setup](#3-environment-setup). It is recommended to perform all subsequent steps in [the provided Docker](#32-docker-version).

## 2. Features
* Supports BM1684X (x86 PCIe, SoC)
* Supports FP16, INT8, and INT4 model compilation and inference
* Supports Python demos based on pybind inference

## 3. Environment Setup
No memory modification is needed on PCIe; the following is for SoC mode:
For 1684X series devices (such as SE7/SM7), you can complete environment preparation in this way to meet the requirements for running Qwen. First, on the 1684X SoC environment, modify the device memory with the following commands.
```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/09/11/13/DeviceMemoryModificationKit.tgz
tar xvf DeviceMemoryModificationKit.tgz
cd DeviceMemoryModificationKit
tar xvf memory_edit_{vx.x}.tar.xz # vx.x is the version number
cd memory_edit
./memory_edit.sh -p # this command prints the current memory layout information
./memory_edit.sh -c -npu 7615 -vpu 3072 -vpp 3072 # npu can also access the memory of vpu and vpp
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/boot.itb /boot/boot.itb && sync
sudo reboot
```
> **Note:**
> 1. The total tpu memory is the sum of npu/vpu/vpp. fp16 models require tpu memory >= 12800 MB, int8 requires tpu memory >= 7168 MB, and int4 requires tpu memory >= 4608 MB.
> 2. For more tutorials, please refer to [SoC Memory Modification Tool](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html)

### 3.1 mlir version
  Get the TPU-MLIR archive from sftp
  ```bash
  pip3 install dfss --upgrade
  python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/tpu-mlir_v1.6.113-g7dc59c81-20240105.tar.gz 
  tar -xf tpu-mlir_v1.6.113-g7dc59c81-20240105.tar.gz 
  ```

### 3.2 docker version
  The docker used by TPU-MLIR is sophgo/tpuc_dev:latest. The docker image and tpu-mlir are bound to each other; in rare cases, tpu-mlir may be updated and require a new image.
  ```bash
  docker pull sophgo/tpuc_dev:latest
  # This maps the current directory to /workspace inside docker; users need to map the demo directory into docker according to their actual situation
  # myname is just an example name; please specify the container name you want
  docker run --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest
  # You are now inside docker, in the /workspace directory
  # Initialize the software environment
  cd /workspace/tpu-mlir_vx.y.z-<hash>-<date>
  source ./envsetup.sh
  ```
For the path of `tpu-mlir_vx.y.z-<hash>-<date>`, use [the mlir path downloaded earlier](#31-mlir-version). For more TPU-MLIR tutorials, please refer to the "TPU-MLIR Quick Start Guide" and the "TPU-MLIR Development Reference Manual" on the [SOPHGO official website](https://developer.sophgo.com/site/index/material/31/all.html).

## 4. Model Compilation
## 4.1 Obtain onnx
### 4.1.1 Download the official Qwen1.5 code

**Note:** The official Qwen1.5-7B repository is about 50 GB. Before downloading, make sure you have an access token or SSH key for the HuggingFace website. (The procedure is the same for Qwen1.5-1.8B / Qwen1.5-14B; make sure the corresponding memory requirements are met.) The following commands use Qwen1.5-7B as an example.

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen1.5-7B-Chat
```
If the process hangs after git clone, try interrupting it with `ctrl+c`, then enter the repository and run `git lfs pull`.

### 4.1.2 Modify the official code:
The `compile/files` directory of this demo provides the modified `config.json` and `modeling_qwen2.py` under the directory corresponding to the model. (Please update transformers to 4.38.2 or above.) You can directly replace the files in the original repository:

```bash
cp files/Qwen1.5-1.8B-Chat/config.json Qwen1.5-1.8B-Chat/
cp files/Qwen1.5-1.8B-Chat/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/
```

### 4.1.3 Export onnx
- Export all onnx models; if you are prompted that certain components are missing during the process, simply **pip install** them
- (<strong>Do not use torch 2.1.1 or above, otherwise SPDA-related errors will occur; torch==2.0.1 and transformers==4.38.2 are recommended</strong>)

```bash
# Replace /workspace/Qwen-7B-Chat with the path of your Qwen-7B-Chat repository in the docker environment
python3 compile/export_onnx.py --model_path /workspace/Qwen1.5-1.8B-Chat --seq_length your_sequence_length
```
At this point, a large number of onnx models are exported to the `compile/tmp/onnx` directory of this demo.

### 4.2 bmodel compilation
First, activate the environment under the mlir tool. If you have not downloaded the mlir tool before, [refer to the mlir download address](./Qwen_Export_Guide.md/#212-下载并解压tpu-mlir)
```bash
cd tpu-mlir_v1.6.113-g7dc59c81-20240105
source envsetup.sh
```
Currently, TPU-MLIR supports BF16 (Qwen-1.8B only), INT8, and INT4 quantization of Qwen1.5 on 1684X. Use the following command to generate the bmodel.

```bash
./compile.sh --mode int4 --name qwen1.5-1.8b
```

Here, mode can be specified as bf16/int8/int4. After successful compilation, the model will be stored in the `compile` directory.

### 4.3 Prepare the tokenizer
The tokenizer has been placed in the `support` folder.

## 5. Model Inference
You can use our pre-compiled model for automated testing, or use your own compiled model for inference.
### 5.1 Automated testing
Run under the Qwen1.5 path (the default test model is Qwen1.5-1.8B)
```bash
./run_demo.sh
```
### 5.2 Test your own model
Compile the model inference dynamic library in python_demo
```bash
cd python_demo
mkdir build && cd build
cmake ..
make -j4
cp chat.cpython-310-x86_64-linux-gnu.so ..
cd ..
```

Now you can run model inference
```bash
source ../../../envsetup.sh # Before model inference, make sure to activate the environment variables with `source envsetup.sh` under the LLM-TPU path.
python chat.py --devid '0' --model_path your_bmodel_path --tokenizer_path ../support/token_config/
```
Replace `your_bmodel_path` with the actual path.

