# Phi-3/Phi-4

This project implements the deployment of the large language models [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) and [Phi-4-awq](https://huggingface.co/stelterlab/phi-4-AWQ) on BM1684X. The models are converted to bmodel through the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed to the PCIE environment or the SoC environment of BM1684X using c++ code.

## Development environment preparation

### 1. Download docker and start the container

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```
The following assumes that all environments are in the `/workspace` directory of docker.

### 2. Download `TPU-MLIR` code and compile

(You can also directly download the pre-compiled release package and unzip it)

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  #activate environment variables
./build.sh #compile mlir
```

## Compile the model

When compiling, just use the one-click compilation command. The generated compilation files are saved in the ./phi3 directory
```shell
llm_convert.py -m /workspace/Phi-3-mini-4k-instruct -s 512 -q w4f16 -g 128 --num_device 1  -c bm1684x  -o phi3
```

If compiling phi4, the files are saved in the ./phi4 directory
```shell
llm_convert.py -m /workspace/phi-4-AWQ -s 512 -q w4f16 -g 128 --num_device 1  -c bm1684x  -o phi4
```

## Run the Demo


### python demo

If running phi4, modify the following in pipeline.py
```
self.EOS = [32000, 32007]
```
to
```
self.EOS = [self.tokenizer.eos_token_id]
```

Then execute the following steps

```
cd python_demo
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..


python3 pipeline.py --model_path phi-xxxx.bmodel --tokenizer_path config --devid 0
```
