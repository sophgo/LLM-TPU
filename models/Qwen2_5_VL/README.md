# Qwen2.5-VL

本工程实现BM1684X部署多模态大模型[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

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
将`compile/files/Qwen2_5-VL-3B-Instruct/`下的文件替换`Qwen2.5-VL-3B-Instruct`中对应的文件。
``` shell
cp files/Qwen2_5-VL-3B-Instruct/modeling_qwen2_5_vl.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
```

#### 2. 导出成onnx模型

如果过程中提示缺少某些组件，直接`pip3 install 组件`即可

``` shell
# 导出onnx
cd compile
python3 export_onnx.py --model_path /path/to/Qwen2_5-VL-3B-Instruct --seq_length 8192
```

## 编译模型

此处介绍如何将onnx模型编译成bmodel。也可以省去编译模型这一步，直接下载编译好的模型：

``` shell
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-vl-3b_w4bf16_seq8192.bmodel
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

对ONNX模型进行编译，生成模型`qwen2.5-vl-3b_w4bf16_seq8192.bmodel `

``` shell
./compile.sh --name qwen2-vl-2b --seq_length 8192
```

## 编译与运行程序

* 环境准备
> （python_demo运行之前都需要执行这个）
``` shell
sudo apt-get update
sudo apt-get install pybind11-dev
pip3 install torchvision pillow qwen_vl_utils transformers --upgrade
```

编译库文件，生成`chat.cpython*.so`文件，将该文件拷贝到`pipeline.py`文件目录

``` shell
cd python_demo_video
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

* python demo

``` shell
python3 pipeline.py --model_path qwen2.5-vl-3b_w4bf16_seq8192.bmodel --config_path ../support/processor_config 
```
model为实际的model储存路径；config_path为配置文件路径

* 运行效果

![](../../assets/qwen2_5vl.png)

## 常见问题

#### 是否支持Qwen2_5-VL-7B ?

是支持的，步骤基本一致。
1. 将`files/Qwen2_5-VL-7B`里面的文件替换到`Qwen2_5-VL-7B`中；
2. 执行`export_onnx.py`指定`Qwen2_5-VL-7B`路径，导出onnx；
3. 执行`./compile.sh --name qwen2_5-vl-7b`生成模型`qwen2.5-vl-3b_w4bf16_seq8192.bmodel`