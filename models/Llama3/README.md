# Llama3

本工程实现BM1684X/BM1688部署大模型[Llama3](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到PCIE环境，或者SoC环境。


本文包括如何编译bmodel，和如何在BM1684X/BM1688环境运行bmodel。编译LLM环节可以省去，直接用以下链接下载：

``` shell
# 1684x 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama-3.2-3b-instruct_w4f16_seq512_bm1684x_1dev_20250526_160605.bmodel
# 1688 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/llama-3.2-3b-instruct_w4f16_seq512_bm1688_2core_20250526_161500.bmodel
```

## 编译LLM模型

此处介绍如何将LLM编译成bmodel。

#### 1. 从Huggingface下载`Llama3-3B`

(比较大，会花费较长时间)

``` shell
# 下载模型
git lfs install
git clone git@hf.co:meta-llama/Llama-3.2-3B-Instruct
```

#### 2. 下载docker，启动容器

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
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
# 如果有提示transformers版本问题，pip3 install transformers --upgrade
# 如果要编译bm1688，只需要-c bm1688即可
llm_convert.py -m /workspace/Llama-3.2-3B-Instruct -s 512 -q w4f16 -g 128 -c bm1684x -o llama3.2_3b
```
编译完成后，在指定目录`llama3.2_3b`生成`llama-3.2-3bxxxx.bmodel`和`config`

## 编译与运行程序

请将程序拷贝到PCIE环境或者SoC环境后再编译。然后把`llama-3.2-3bxxxx.bmodel`和`config`拷贝过去。

#### python demo

编译库文件，生成`chat.cpython*.so`文件，将该文件拷贝到`pipeline.py`文件目录

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

* python demo

``` shell
python3 pipeline.py -m llama3_xxx.bmodel -c config 
```
model为实际的model储存路径；config为配置文件路径
