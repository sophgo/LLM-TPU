# Qwen2-VL

本工程实现BM1684X/BM1688部署多模态大模型[Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-AWQ)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

## 开发环境准备

#### 1. 从Huggingface下载`Qwen2-VL-2B-Instruct-AWQ`

(比较大，会花费较长时间)

``` shell
git lfs install
git clone git@hf.co:Qwen/Qwen2-VL-2B-Instruct-AWQ
```

## 编译模型

此处介绍如何将onnx模型编译成bmodel。也可以省去编译模型这一步，直接下载编译好的模型：

``` shell
# 1684x, 672x896
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2-vl-2b-instruct-awq_w4bf16_seq2048_bm1684x_1dev_20250616_180313.bmodel

# 1688, 672x896
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2-vl-2b-instruct-awq_w4bf16_seq2048_bm1688_2core_20250616_193802.bmodel
```

#### 1. 下载docker，启动容器

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```
后文假定环境都在docker的`/workspace`目录。

#### 2. 下载`TPU-MLIR`代码并编译

(也可以直接下载编译好的release包解压)

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  #激活环境变量
./build.sh #编译mlir
```

#### 3. 编译模型生成bmodel

``` shell
# 如果有提示transformers版本问题，pip3 install transformers -U
# bm1688 需要指定-c bm1688; -g 指定group size; AWQ和GPTQ实际上是不需要指定group_size的；非AWQ/GPTQ模型是需要指定
llm_convert.py -m /workspace/Qwen2-VL-2B-Instruct-AWQ -s 2048 --quantize w4f16 -c bm1684x --out_dir qwen2vl_2b --max_pixels 672,896
```
编译完成后，在指定目录`qwen2vl_2b`生成`qwen2vl_2b-xxx.bmodel`和`config`

## 编译与运行程序

编译库文件，生成`chat.cpython*.so`文件，将该文件拷贝到`pipeline.py`文件目录;
并将bmodel文件和config目录拷贝过去

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

* python demo

``` shell
python3 pipeline.py -m qwen2-vl-2b-xxxx.bmodel -c config
```
