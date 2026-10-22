# MiniCPM-V-4.7

This project deploys the multimodal models [MiniCPM-V-4.7-1B-AWQ](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-AWQ) and [MiniCPM-V-4.7-35B-A3B-AWQ](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-AWQ) on BM1684X/CV84X6. Each model is converted into a bmodel through the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler and deployed to the PCIE environment or SoC environment.

Both models support image and video recognition. A Python demo is provided and is shared by the two variants; use the bmodel and the `config` directory produced by the same compile.

## Compile the LLM Model

This section describes how to compile the model into a bmodel.

#### 1. Download the Model

``` shell
git lfs install
# 1B
git clone https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-AWQ
# 35B-A3B
git clone https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-AWQ
# make sure the safetensors are fully downloaded (git lfs pull), not LFS pointer stubs
```

#### 2. Download docker and Start the Container

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
The following assumes that the environment is in the `/workspace` directory of docker.

#### 3. Download the `TPU-MLIR` Code and Compile It

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  # activate the environment variables
./build.sh # compile mlir
```

#### 4. Compile the Model to Generate the bmodel

``` shell
# max_input_length specifies the maximum input length; if not specified, the length specified by -s is used
llm_convert.py -m /workspace/MiniCPM-V-4.7-1B-AWQ -s 2048 --max_input_length 1024 -q auto -c bm1684x -o minicpm_v4_7_1b
```
After compilation, `minicpm-v-4.7-xxx.bmodel` and `config` are generated in the specified directory.

## Compile and Run the Program (python)

### 1. Environment Preparation

A python3.10 environment is required. If it is not available, refer to [this document](https://github.com/sophgo/sophon-demo/blob/release/docs/FAQ.md#13-se7%E5%AE%89%E8%A3%85python310) to install it.

``` shell
sudo apt-get update
sudo apt-get install pybind11-dev

pip3 install torch==2.6.0 torchvision==0.21.0 transformers==5.18.0
```

### 2. Compile the Library Files

Compile the C++ library files to generate `chat.cpython*.so`:

``` shell
cd python_demo
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

### 3. Run

``` shell
# Interactive mode
python3 pipeline.py -m minicpm-v-4.7.bmodel -c ../config

# Single inference mode (image)
python3 pipeline.py -m minicpm-v-4.7.bmodel -c ../config \
    --prompt "Describe this image @test.jpg"

# Single inference mode (video)
python3 pipeline.py -m minicpm-v-4.7.bmodel -c ../config \
    --prompt "Describe what happened in the video @test.mp4"
```
