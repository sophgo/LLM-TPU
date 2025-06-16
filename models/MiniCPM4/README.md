# MiniCPM4

本工程实现BM1684X/BM1688部署大模型[MiniCPM4](https://huggingface.co/openbmb/MiniCPM4-8B)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到PCIE环境，或者SoC环境。


本文包括如何编译bmodel，和如何在BM1684X/BM1688环境运行bmodel。编译LLM环节可以省去，直接用以下链接下载：

``` shell
# minicpm4-8b 1684x 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm4-8b_w4bf16_seq512_bm1684x_1dev_20250613_175044.bmodel
# minicpm4-8b 1684x 8k, 动态模型
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm4-8b_w4bf16_seq8192_bm1684x_1dev_20250613_182940.bmodel

# minicpm4-0.5b bm1688 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/minicpm4-0.5b-gptq_w4bf16_seq512_bm1688_2core_20250616_122001.bmodel
# minicpm4-0.5b cv186x 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/minicpm4-0.5b-gptq_w4bf16_seq512_cv186x_1core_20250616_122126.bmodel

```

## 编译LLM模型

此处介绍如何将LLM编译成bmodel。

#### 1. 从Huggingface下载MiniCPM4

(比较大，会花费较长时间)

``` shell
# 下载模型
git lfs install
git clone git@hf.co:openbmb/MiniCPM4-0.5B
# 如果是8B，则如下：
git clone git@hf.co:openbmb/MiniCPM4-8B
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
llm_convert.py -m /workspace/MiniCPM4-8B -s 512 --quantize w4bf16 -c bm1684x --out_dir minicpm4-8b
```
编译完成后，在指定目录`minicpm4-8b`生成`minicpm4-xxx.bmodel`和`config`

另外如果指定的seqlen比较长的话，比如8K，可以指定`--dynamic`编译，首token延时会根据实际长度变化，如下：
``` shell
# 如果有提示transformers版本问题，pip3 install transformers --upgrade
llm_convert.py -m /workspace/MiniCPM4-8B -s 8192 --quantize w4bf16 -c bm1684x --dynamic --out_dir minicpm4-8b
```

## 编译与运行程序

请将程序拷贝到PCIE环境或者SoC环境后再编译。然后把`minicpm4-xxx.bmodel`和`config`拷贝过去。

#### python demo

编译库文件，生成`chat.cpython*.so`文件，将该文件拷贝到`pipeline.py`文件目录

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

* python demo

``` shell
python3 pipeline.py -m minicpm4_xxx.bmodel -c config 
```
model为实际的model储存路径；config为配置文件路径
