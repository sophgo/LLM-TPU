# Qwen2.5-VL

本工程实现BM1684X部署多模态大模型[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。


本文包括如何转换bmodel，和如何在BM1684X环境运行bmodel。如何转换bmodel环节可以省去，直接用以下链接下载：

``` shell
# 3B 2K版本
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-vl-3b_w4bf16_seq2048.bmodel
# 3B 8K版本
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-vl-3b_w4bf16_seq8192.bmodel
# 7B 2K版本
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-vl-7b_w4bf16_seq2048.bmodel
# 7B 8K版本
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-vl-7b_w4bf16_seq8192.bmodel
```

## 开发环境准备

#### 1. 从Huggingface下载`Qwen2.5-VL-3B-Instruct`

(比较大，会花费较长时间)

``` shell
# 安装依赖的组件
pip3 install qwen-vl-utils accelerate torch torchvision transformers
# 下载模型
git lfs install
git clone git@hf.co:Qwen/Qwen2.5-VL-3B-Instruct
```

另外需要做一些模型源码上的修改：
1. 修改`Qwen2_5-VL-3B-Instruct`的`config.json`中的`max_position_embeddings`改成想要的长度，比如2048
2. 将`compile/files/Qwen2_5-VL/`中的`modeling_qwen2_5_vl.py`覆盖到transformers中，比如：
``` shell
cp files/Qwen2_5-VL/modeling_qwen2_5_vl.py /root/miniconda3/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
```

#### 2. 导出成onnx模型

如果过程中提示缺少某些组件，直接`pip3 install 组件`即可

``` shell
# 导出onnx
cd compile
python3 export_onnx.py --model_path /workspace/Qwen2.5-VL-3B-Instruct --seq_length 2048 --device cuda
```

## 编译模型

此处介绍如何将onnx模型编译成bmodel。

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

对ONNX模型进行编译，生成模型`qwen2.5-vl-3b_w4bf16_seq2048.bmodel `

``` shell
cd compile
./compile.sh --name qwen2.5-vl-3b --seq_length 2048
```

## 编译与运行程序

* 环境准备
> （python_demo运行之前都需要执行这个）
``` shell
# 如果不是python3.10，参考"常见问题"配置环境
pip3 install torchvision pillow qwen_vl_utils transformers>=4.49.0
```

编译库文件，生成`chat.cpython*.so`文件，将该文件拷贝到`pipeline.py`文件目录

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

* python demo

``` shell
python3 pipeline.py --model_path qwen2.5-vl-3b_w4bf16_seq2048.bmodel --config_path config 
```
model为实际的model储存路径；config_path为配置文件路径

* 运行效果

![](../../assets/qwen2_5vl.png)

## 常见问题

#### 本实例模型如何确保INT4的精度？

为了提高精度，本示例中的模型是从AWQ模型反量化而来。参考项目[llmc-tpu](https://github.com/sophgo/llmc-tpu)，
在该项目的docker中，经过如下命令转换：
``` shell
python3 tpu/llm_dequant.py --pretrained_model_path /workspace/Qwen2.5-VL-3B-Instruct --quant_model_path /workspace/Qwen2.5-VL-3B-Instruct-AWQ --model_type qwen2.5_vl --use_cpu
```

#### SoC如何配置python3.10环境 ?

安装过程如下：

``` shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-dev
```

python虚拟环境配置：

``` shell
cd /data
# 创建虚拟环境（不包含 pip）
python3.10 -m venv --without-pip myenv

# 进入虚拟环境
source myenv/bin/activate

# 手动安装 pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
rm get-pip.py

# 安装依赖库
pip3 install torchvision pillow qwen_vl_utils transformers --upgrade

```

#### 是否支持Qwen2_5-VL-7B ?

是支持的，步骤基本一致。
1. 将`files/Qwen2_5-VL`里面的`model_qwen2_5_vl.py`替换到transformers中；
2. 将`Qwen2_5-VL-7B-Instruct`中`config.json`配置max_position_embeddings改为2048;
3. 执行`export_onnx.py`指定`Qwen2_5-VL-7B-Instruct`路径，导出onnx；
4. 执行`./compile.sh --name qwen2_5-vl-7b --seq_length 2048`生成模型`qwen2.5-vl-7b_w4bf16_seq2048.bmodel`