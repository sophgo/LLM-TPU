# Gemma3

本工程实现BM1684X/BM1688部署大模型[Gemma3](https://huggingface.co/google/gemma-3-4b-it)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到PCIE环境，或者SoC环境。


## 编译LLM模型

此处介绍如何将LLM编译成bmodel。

#### 1. 从Huggingface下载`Gemma3-4B`

(比较大，会花费较长时间)

``` shell
git lfs install
git clone git@hf.co:google/gemma-3-4b-it
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
llm_convert.py -m /workspace/gemma-3-4b-it -s 2048 --quantize w4bf16 -c bm1684x --out_dir gemma3_4b
```
编译完成后，在指定目录`gemma3_4b`生成`gemma3-xxx.bmodel`和`config`



## 编译与运行程序

请将程序拷贝到PCIE环境或者SoC环境后再编译。然后把`gemma3-xxx.bmodel`和`config`拷贝过去。

#### python demo

编译库文件，生成`chat.cpython*.so`文件，将该文件拷贝到`pipeline.py`文件目录

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

* python demo

``` shell
python3 pipeline.py -m gemma3_xxx.bmodel -c config 
```
model为实际的model储存路径；config为配置文件路径
