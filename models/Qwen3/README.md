# Qwen3

本工程实现BM1684X/BM1688部署大模型[Qwen3](https://huggingface.co/Qwen/Qwen3-4B-AWQ)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到PCIE环境，或者SoC环境。


本文包括如何编译bmodel，和如何在BM1684X/BM1688环境运行bmodel。编译LLM环节可以省去，直接用以下链接下载：

``` shell
# 1684x 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-4b-awq_w4bf16_seq512_bm1684x_1dev_20250514_161445.bmodel
# 1684x 8k, 静态模型
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-4b-awq_w4bf16_seq8192_bm1684x_1dev_20250514_161732.bmodel
# 1684x 8k, 动态模型
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-4b-awq_w4bf16_seq8192_bm1684x_dyn.bmodel

# 1688 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/qwen3-4b-awq_w4bf16_seq512_bm1688_2core_20250514_162231.bmodel
```

### deepseek-r1-0528-qwen3-8b

``` shell
# 1684x 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-0528-qwen3-8b_w4bf16_seq512_bm1684x.tar
# 1684x 4k
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-0528-qwen3-8b_w4bf16_seq4096_bm1684x.tar
```

## 编译LLM模型

此处介绍如何将LLM编译成bmodel。

#### 1. 从Huggingface下载`Qwen3-4B-AWQ`

(比较大，会花费较长时间)

``` shell
# 下载模型
git lfs install
git clone git@hf.co:Qwen/Qwen3-4B-AWQ
# 如果是8B，则如下：
git clone git@hf.co:Qwen/Qwen3-8B-AWQ
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
llm_convert.py -m /workspace/Qwen3-4B-AWQ -s 512 --quantize w4bf16 -c bm1684x --out_dir qwen3_4b
```
编译完成后，在指定目录`qwen3_4b`生成`qwen3-xxx.bmodel`和`config`

另外如果指定的seqlen比较长的话，比如8K，可以指定`--dynamic`编译，首token延时会根据实际长度变化，如下：
``` shell
# 如果有提示transformers版本问题，pip3 install transformers --upgrade
llm_convert.py -m /workspace/Qwen3-4B-AWQ -s 8192 --quantize w4bf16 -c bm1684x --dynamic --out_dir qwen3_4b
```

## 编译与运行程序

请将程序拷贝到PCIE环境或者SoC环境后再编译。然后把`qwen3-xxx.bmodel`和`config`拷贝过去。

#### python demo

编译库文件，生成`chat.cpython*.so`文件，将该文件拷贝到`pipeline.py`文件目录

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

* python demo

``` shell
python3 pipeline.py -m qwen_xxx.bmodel -c config
```
model为实际的bmodel储存路径；config为编译模型时生成的配置文件，demo中存放了qwen3的config。
* 如果更换其他系列模型，如deepseek-r1-distill-qwen，需要指定新的config

#### cpp demo

``` shell
mkdir -p build
cd build
cmake .. && make && cd ..

# how to run
./qwen3 -m qwen3_xxx.bmodel -c config
```