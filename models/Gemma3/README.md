# Gemma3

This project implements the deployment of the large model [Gemma3](https://huggingface.co/google/gemma-3-4b-it) on BM1684X/BM1688. The model is converted to bmodel through the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed to the PCIE environment or the SoC environment using c++ code.


## Compile the LLM model

This section describes how to compile the LLM into bmodel.

#### 1. Download `Gemma3-4B` from HuggingFace

(Relatively large, will take a long time)

``` shell
git lfs install
git clone git@hf.co:google/gemma-3-4b-it
```

#### 2. Download docker and start the container

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
The following assumes that all environments are in the `/workspace` directory of docker.

#### 2. Download `TPU-MLIR` code and compile

(You can also directly download the pre-compiled release package and unzip it)

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  #activate environment variables
./build.sh #compile mlir
```

#### 3. Compile the model to generate bmodel

``` shell
# If prompted about transformers version issues, pip3 install transformers -U
llm_convert.py -m /workspace/gemma-3-4b-it -s 2048 --quantize w4bf16 -c bm1684x --out_dir gemma3_4b
```
After compilation is complete, `gemma3-xxx.bmodel` and `config` are generated in the specified directory `gemma3_4b`.



## Compile and run the program

Please copy the program to the PCIE environment or the SoC environment before compiling. Then copy `gemma3-xxx.bmodel` and `config` over as well.

#### python demo

Compile the library file to generate the `chat.cpython*.so` file, and copy this file to the directory where `pipeline.py` is located

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

* python demo

``` shell
python3 pipeline.py -m gemma3_xxx.bmodel -c config 
```
model is the actual model storage path; config is the configuration file path
