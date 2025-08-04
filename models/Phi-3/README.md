# Phi-3/Phi-4

本工程实现BM1684X部署语言大模型[Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)和[Phi-4-awq](https://huggingface.co/stelterlab/phi-4-AWQ)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

## 开发环境准备

### 1. 下载docker，启动容器

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```
后文假定环境都在docker的`/workspace`目录。

### 2. 下载`TPU-MLIR`代码并编译

(也可以直接下载编译好的release包解压)

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  #激活环境变量
./build.sh #编译mlir
```

## 编译模型

编译时采用一键编译指令即可，生成的编译文件保存在 ./phi3 目录中
```shell
llm_convert.py -m /workspace/Phi-3-mini-4k-instruct -s 512 -q w4f16 -g 128 --num_device 1  -c bm1684x  -o phi3
```

如果编译phi4则保存在 ./phi4 目录中
```shell
llm_convert.py -m /workspace/phi-4-AWQ -s 512 -q w4f16 -g 128 --num_device 1  -c bm1684x  -o phi4
```

## 运行Demo


### python demo

如果运行phi4则将 pipeline.py 中的
```
self.EOS = [32000, 32007]
```
修改为
```
self.EOS = [self.tokenizer.eos_token_id]
```

再执行以下步骤

```
cd python_demo
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..


python3 pipeline.py --model_path phi-xxxx.bmodel --tokenizer_path config --devid 0
```
