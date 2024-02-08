![](./assets/sophgo_chip.png)

# ChatGLM3-TPU

本项目实现BM1684X部署语言大模型[ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。


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


2. 下载`ChatGLM3-6B`，比较大，会花较长时间

``` shell
git lfs install
git clone git@hf.co:THUDM/chatglm3-6b
```

并对该工程做三点修改：
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

3. 下载`TPU-MLIR`代码并编译，(也可以直接下载编译好的release包解压)

``` shell
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```

4. 下载本项目`ChatGLM3-TPU`，如下：

``` shell
git clone git@github.com:sophgo/ChatGLM3-TPU.git
```

## 编译模型

1. 指定`ChatGLM3-6B`的python路径

``` shell
export PYTHONPATH=/workspace/chatglm3-6b:$PYTHONPATH
```

2. 导出所有onnx模型，如果过程中提示缺少某些组件，直接`pip install 组件`即可

``` shell
cd ChatGLM3-TPU/compile
python3 export_onnx.py
```
此时有大量onnx模型被导出到tmp目录。

3. 对onnx模型进行编译

目前TPU-MLIR支持对ChatGLM3进行F16、INT8和INT4量化，且支持多芯分布式推理，默认情况下会进行F16量化和单芯推理，最终生成`chatglm3-6b.bmodel`文件

```shell
./compile.sh
```

若想进行INT8或INT4量化，则执行以下命令，最终生成`chatglm3-6b_int8.bmodel`或`chatglm3-6b_int4.bmodel`文件，如下命令：

```shell
./compile.sh --mode int8 # or int4
```

若想进行2芯推理，则执行以下命令，最终生成`chatglm3-6b_f16_2dev.bmodel`文件，4芯8芯同理：

```shell
./compile.sh --num_device 2
```

## 编译程序(C++版本)

执行如下编译，默认是PCIE版本：

```shell
cd ChatGLM3-TPU/demo
mkdir build
cd build
cmake ..
make
```

如果是SoC版本，有两种编译方法：

方法1：直接将demo目录拷贝到SoC环境，按以上步骤编译(推荐)

方法2：docker中交叉编译，如下操作

```shell
wget https://releases.linaro.org/components/toolchain/binaries/7.5-2019.12/aarch64-linux-gnu/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz
tar -xvf gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz
mv gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu /opt/aarch64-linux-gnu-7.5.0
cd ChatGLM3-TPU/demo
mkdir build
cd build
cmake .. -DTARGET_ARCH=soc
make -j

```

编译生成chatglm可执行程序，将`chatglm`放到/ChatGLM3-TPU/demo目录下，同时按照下列方式指定芯片数量和bmodel路径。
运行`chatglm`，默认单芯运行`chatglm3-6b.bmodel`:
```shell
./chatglm --model chatglm3-6b.bmodel
```

如果是要运行INT8或INT4模型，则命令如下：
```shell
./chatglm --model chatglm3-6b_int8.bmodel # same with int4
```

如果是2芯分布式推理，使用如下命令(比如指定在2号和3号芯片上运行, 用`source /etc/profiel`后使用`bm-smi`查询芯片id号)：
```shell
./chatglm --model chatglm3-6b_f16_2dev.bmodel --devid 2,3
```

## 编译程序(Python Web版本)

```shell
pip install gradio==3.39.0
cd ChatGLM3-TPU/web_demo
mkdir build
cd build
cmake ..
make -j
```

编译成功会生成`libtpuchat.so*`, 在web_demo.py中指定bmodel\_path token\_path device\_id, lib_path(编译生产的.so文件), 以及dev_id。
```python
python web_demo.py --dev 0 --bmodel_path your_bmodel_path
```
即可成功运行web的demo。

如果是SoC环境，参考C++版本

PS：尽量下载gradio==3.39.0版本，不然会出现各种问题！！

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
原因是运行环境的库有所不同，将demo中的`lib_pcie`（PCIE）或者 `lib_soc`(SoC)里面的so文件拷贝到运行环境，链接到里面的so即可。


## 工具调用
参考：[工具调用](./tools_using/README.md)