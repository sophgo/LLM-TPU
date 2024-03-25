![image](./assets/sophgo_chip.png)

# Llama2

本项目实现BM1684X部署语言大模型[Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

下文中默认是PCIE环境；如果是SoC环境，按提示操作即可。

# 目录说明
```
.
├── README.md                            #使用说明
├── requirements.txt                     #需要使用的python wheel包
├── demo                                 #Llama2 c++代码文件
│   ├── CMakeLists.txt
│   └── demo.cpp                         #主程序
├── web_demo                             #Llama2 web demo代码文件
│   ├── CMakeLists.txt
│   ├── chat.cpp                         #cpp主程序
│   ├── chat.py                          #pybind 后的python主程序
│   └── web_demo.py                      #gradio python界面代码
├── compile
│   ├── compile.sh                       #用来编译TPU模型的脚本
│   └── export_onnx.py                   #用来导出onnx的脚本
│       └── files                        #用于替换原模型的文件
└── support
    ├── include                          #编译所需的库文件
    ├── lib_pcie                         #编译PCIE版本所需头文件
    ├── lib_soc                          #编译SOC版本所需头文件
    └── tokenizer.model                  #分词模型
```
----------------------------

# 【阶段一】模型编译

## 注意点
* 模型编译必须要在docker内完成，无法在docker外操作

### 步骤一：模型下载
虽然Llama2模型允许商业开源，但是模型下载需要想Meta提交使用申请，因此测试模型时可以使用我们已经下载好的模型
```bash
pip3 install dfss
# llama2-7B
python3 -m dfss --url=open@sophgo.com:sophon-demo/Llama2/llama2-7b-torch.zip
unzip llama2-7b-torch.zip

# llama2-13B
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/llama2-13b-torch.zip
unzip llama2-13b-torch.zip
```

### 步骤二：下载docker

下载docker，启动容器，如下：

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

### 步骤三：下载TPU-MLIR代码并编译

``` shell
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```
* PS：重新进入docker环境并且需要编译模型时，必须在此路径下执行上述`source ./envsetup.sh` 和 `./build.sh`才能完成后续模型编译。

### 步骤四：对齐模型环境

``` shell
pip install -r requirements.txt
cp ./compile/files/llama-2-7b-chat-hf/modeling_llama.py /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
```

* PS：不一定是/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py这个路径，建议替换前先pip show transformers查看一下

### 步骤五：生成onnx文件

``` shell
cd compile
python export_onnx.py --model_path your_model_path --seq_length 512
```

* PS1：your_model_path 指的是原模型下载后的地址, 如:"../../torch2onnx/llama-2-7b-chat-hf", 可以根据需要选择使用7b模型还是13b模型。
* PS2：如果你想要debug，而不是一下子生成完成全部的onnx模型，可以将240行的num_layers改成1, 结合233行的函数对比单个block情况下是否可以和
* PS3：默认导出sequence length为512的模型

### 步骤六：生成bmodel文件

生成单芯模型

``` shell
./compile.sh --mode int8 --name llama2-7b # same as int4
```

生成双芯模型

``` shell
./compile.sh --mode int8 --num_device 2 --name llama2-7b # same as int4
```

* PS1：编译完成后最终会在compile路径下生成名为llama2-{X}b_{Y}_{Z}dev.bmodel,其中X为7或13，Y为`compile.sh`时选择的`mode`的数据类型,Z为推理的芯片数量(如果不指定num_device, 会省略{Z}dev的部分)
* PS2：生成bmodel耗时大概3小时以上，建议64G内存以及200G以上硬盘空间，不然很可能OOM或者no space left
* PS3：如果想要编译llama2-7b，则--name必须为llama2-7b，想要编译llama2-13b，则必须--name为llama2-13b
* PS4：目前给定的lib_pcie和lib_soc部分仅包含单芯的动态库，多芯部分会在后续更新

----------------------------

# 阶段二：可执行文件生成

## 编译程序(C++版本)

执行如下编译，(PCIE版本与SoC版本相同)：

```shell
cd demo
mkdir build
cd build
cmake ..
make
```

编译生成llama2可执行程序，将`llama2`放到demo目录下，同时按照下列方式指定芯片编号（默认只使用0号芯片）和bmodel路径。运行`llama2`:
```shell
./llama2 --model your_llama2_bmodel_path --tokenizer ../support/tokenizer.model --dev dev_id
```

如果是双芯分布式推理，使用如下命令(比如指定在2号和3号芯片上运行, 用`source /etc/profiel`后使用`bm-smi`查询芯片id号,查看需要在之前安装过libsophon驱动)：
```shell
./llama2 --model your_llama2_bmodel_path --tokenizer ../support/tokenizer.mode --devid 2,3
```
* PS：请勿将编译好的单芯模型用多颗芯片进行推理。可以在编译好的bmodel名称中了解它是否是多芯模型，如`llama2-7b_int8_2dev.bmodel`是可以跑双芯的模型。双芯模型可以用单芯运行。

## 编译程序(Python Web版本)【单芯】

```shell
pip install gradio==3.39.0
cd web_demo
mkdir build
cd build
cmake ..
make -j
```

编译成功会在`build`文件夹下生成`libtpuchat.so*`, 此时可以在web_demo.py中指定bmodel\_path token\_path device\_id, lib_path(编译生产的`libtpuchat.so*`文件, 默认路径是`./build`下), 以及dev_id。
```python
python web_demo.py --dev 0 --bmodel_path your_bmodel_path
```
即可成功运行web的demo。
* PS：在用户不修改上述token\_path的lib\_path的存放路径前提下只需指定bmodel\_path即可运行程序。

如果是SoC环境，参考C++版本

* PS：尽量下载gradio==3.39.0版本，不然会出现各种问题！！

# 常见问题

### 如何跑通Llama2-13B 6芯模型

首先按照[这个链接](https://github.com/sophgo/LLM-TPU/tree/main)中的版本检查，检查sophon-driver版本是否是0.5.0，如果是0.4.9的话会挂死

```shell
cd /workspace/LLM-TPU/models/Llama2/demo
mkdir build && cd build
cmake .. && make && cp llama2 .. && cd ..

python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama2-13b_int4_6dev.bmodel

./llama2 --model llama2-13b_int4_6dev.bmodel --tokenizer ../support/tokenizer.model  --devid 0,1,2,3,4,5


python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama2-13b_int8_6dev.bmodel

./llama2 --model llama2-13b_int8_6dev.bmodel --tokenizer ../support/tokenizer.model  --devid 0,1,2,3,4,5
```


