![](./assets/sophgo_chip.png)

# ChatGLM2

本项目实现BM1684X部署语言大模型[ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。


在知乎上写了关于`ChatGLM`的解读，方便大家理解源码：

[ChatGLM2流程解析与TPU-MLIR部署](https://zhuanlan.zhihu.com/p/641975976)


## 开发环境


1. 下载docker，启动容器，如下：

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
后文假定环境都在docker的`/workspace`目录。


2. 从Huggingface下载`ChatGLM2-6B`，比较大，会花较长时间

``` shell
git lfs install
git clone git@hf.co:THUDM/chatglm2-6b
```
并将本项目中./models/ChatGLM2/compile/files/chatglm2-6b中config.json与modeling_chatglm.py替换至上述下载后的文件夹中，并替换同名文件（其中需要采用其它sequence length的用户请参考[常见问题](#常见问题),默认sequence length = 512）

3. 下载`TPU-MLIR`代码并编译，(也可以直接下载编译好的release包解压)

目前由于mlir还在维护中，编译GLM系列模型的用户请下载
``` shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/mlir_club/glm_mlir.tar.gz
tar -xf glm_mlir.tar.gz
source source tpu-mlir_v1.6.45-gdc3e9f6b-20231220/envsetup.sh 
```

后续mlir维护完成后可以使用如下方式
``` shell
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```

## 编译模型

1. 导出所有onnx模型，如果过程中提示缺少某些组件，直接`pip3 install 组件`即可

``` shell
cd compile
python3 export_onnx.py --model_path your_chatglm2-6b_path
```
此时有大量onnx模型被导出到tmp目录。

2. 对onnx模型进行编译

目前TPU-MLIR支持对ChatGLM2进行F16、INT8和INT4量化，且支持多芯分布式推理，默认情况下会进行F16量化和单芯推理，最终生成`chatglm2-6b_f16_1dev.bmodel`文件

```shell
./compile.sh --name chatglm2-6b --mode inference_mode --num_device device_number
```

其中：
`--name` 为模型名称，在此指定为`chatglm2-6b`；
`--mode` 为推理所使用的数据类型，可以选择`f16, int8, int4`中任意一种，默认为`f16`；
`--num_device` 为推理所使用的芯片数量，请根据实际所使用的设备指定，默认`--num_device 1`。

## 编译程序(C++版本)

执行如下编译，（PCIE与SOC相同）：

```shell
cd demo
mkdir build
cd build
cmake ..
make
```

编译生成chatglm可执行程序，将`chatglm`放到demo目录下，同时按照下列方式指定芯片数量和bmodel路径。
运行`chatglm`，默认单芯运行`chatglm2-6b_f16_1dev.bmodel`:
```shell
./chatglm --model chatglm2-6b_f16_1dev.bmodel --tokenizer ../support/tokenizer/tokenizer.model --devid  your_devid
```
其中`--devid`为用来推理的TPU编号，默认为0，如果使用多芯推理（需要保证编译的bmodel也是多芯）可以使用`,`来增加芯片，如`--devid 2,3` 表示使用TPU2 和 TPU3来进行推理。

## 运行效果

以下为单芯片下INT8量化模式的运行效果：

![](./assets/chatglm.jpg)

## 常见问题

#### sentencepiece是怎么来的

工程中已经有编译好的，所以不需要编译，如果好奇的话，参考如下步骤。

下载[sentencepiece](https://github.com/google/sentencepiece)，并编译得到`libsentencepiece.a`

```shell
git clone git@github.com:google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j
```

如果要编译SoC环境，则参考demo的编译方式，在makefile中指定交叉编译器

#### demo程序无法正常运行

如果demo程序拷贝到运行环境提示无法运行，比如接口找不到等等错误。
原因是运行环境的库有所不同，将demo中的`./support/lib_pcie`（PCIE）或者 `./support/lib_soc`(SoC)里面的so文件拷贝到运行环境，链接到里面的so即可。


#### 对源码做了哪些修改：

一共做了三点修改：
- 将`config.json`文件中`seq_length`配置为512；
- 将`modeling_chatglm.py`文件中的如下代码：

```python
if attention_mask is not None:
    attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
```

修改为：

```python
if attention_mask is not None:
    attention_scores = attention_scores + (attention_mask * -10000.0)
```

这样修改可以提升效率，使用`masked_fill`效率低下；另一方面`masked_fill`转ONNX存在些bug。

- 将`modeling_chatglm.py`文件中的如下代码：

```python
pytorch_major_version = int(torch.__version__.split('.')[0])
if pytorch_major_version >= 2:
```

修改为：

```python
pytorch_major_version = int(torch.__version__.split('.')[0])
if False:
```

这是因为ONNX无法支持`torch.nn.functional.scaled_dot_product_attention`算子的转换。