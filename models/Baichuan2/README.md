![image](../assets/sophgo_chip.png)

# Baichuan2-TPU

本项目实现BM1684X部署语言大模型[Baichuan2-7B](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

下文中默认是PCIE环境；如果是SoC环境，按提示操作即可。

# 目录说明
```
.
├── README.md                           #使用说明
├── requirements.txt                    #需要使用的python wheel包
├── compile
│   ├── compile.sh                      #用来编译TPU模型的脚本
│   ├── export_onnx.py                  #用来导出onnx的脚本
│   ├── torch_inference.py              #torch推理脚本
│   └── files
│       └── Baichuan2-7B                #替换Baichuan2-7B-chat的对应文件的备份
│           ├── config.json
│           └── modeling_baichuan.py
├── demo                                #Baichuan2 c++代码文件
│   ├── CMakeLists.txt
│   └── demo.cpp                        #主程序
├── src                                 #编译依赖库
│   ├── include
│   ├── lib_pcie
│   └── lib_soc
├── model                               #模型文件（bmodel需下载）
│   ├── baichuan2-7b-test_int8.bmodel
│   └── tokenizer.model
└── web_demo                            #web demo，提供网页对话示例
    ├── chat.cpp
    ├── chat.py
    ├── CMakeLists.txt
    └── web_demo.py
```
----------------------------

# 【阶段一】模型编译

## 注意点
* 模型编译必须要在docker内完成，无法在docker外操作

### 步骤一：模型下载
Baichuan2模型在hugging face上完全开源，供用户下载使用。请根据官网下载步骤进行模型与权重的下载。
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat

# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
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

### 步骤四：下载本项目，安装requirements.txt
下载transfomers、sentencepiece、Baichuan2-TPU以及百度网盘里的.bin模型，并替换transformers里面的modeling_baichuan.py

``` shell
git clone https://github.com/sophgo/Baichuan2-TPU.git
cd Baichuan2
pip3 install -r requirements.txt
```

### 步骤五：替换modeling_baichuan.py, 修改config.json, 生成onnx文件
修改Baichuan2-7B-chat项目中config.json文件中max_position_embeddings与model_max_length，从4096变为512

``` shell
cd compile
cp files/Baichuan2-7B/modeling_baichuan.py $BAICHUAN2_PATH
cp files/Baichuan2-7B/config.json $BAICHUAN2_PATH
python3 export_onnx.py --model_path $BAICHUAN2_PATH
```

* PS1：your_model_path 指的是原模型下载后的地址, 如:"../../torch2onnx/Baichuan2-7B-Chat", 可以根据需要选择使用7b模型还是13b模型。
* PS2：如果你想要debug，而不是一下子生成完成全部的onnx模型，可以将240行的num_layers改成1, 并结合函数对比单个block情况下是否可以和

### 步骤六：生成bmodel文件

生成模型

``` shell
./compile.sh --mode int8
mv baichuan2-7b_int8_1dev.bmodel ../model
```

* PS1：编译完成后最终会在Baichuan2-TPU/compile路径下生成名为baichuan2-{X}b_{Y}_{Z}dev.bmodel,其中X为7或13，Y为`compile.sh`时选择的`mode`的数据类型,Z为推理的芯片数量(如果不指定num_device, 会省略{Z}dev的部分)
* PS2：生成bmodel耗时大概3小时以上，建议64G内存以及200G以上硬盘空间，不然很可能OOM或者no space left
* PS3：目前给定的lib_pcie和lib_soc部分仅包含单芯的动态库，多芯部分会在后续更新

----------------------------

# 阶段二：可执行文件生成（可以跳过）

## 准备
* bmodel模型准备：经过阶段一后将得到编译好的bmodel文件【也可以使用我们提供的现成编译好的bmodel文件】，下载方式为:
```shell
cd Baichuan2-TPU/model
pip3 install dfss
# baichuan2-7B
python3 -m dfss --url=open@sophgo.com:sophon-demo/baichuan2/baichuan2-7b-test_int8.bmodel
```
将得到编译好的int8单芯bmodel模型文件。

## 编译程序(C++版本)

执行如下编译，默认是PCIE版本：

```shell
cd Baichuan2-TPU/demo
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
cd Baichuan2-TPU/demo
mkdir build
cd build
cmake .. -DTARGET_ARCH=soc # soc 只有一颗芯片，因此不支持多芯编译
make -j
```

编译生成Baichuan2可执行程序。

运行`baichuan2`:
```shell
./baichuan2 --model ../model/baichuan2-7b-test_int8.bmodel --dev dev_id
```

## 编译程序(Python Web版本)【单芯】

```shell
pip3 install gradio==3.39.0
cd Baichuan2-TPU/web_demo
mkdir build
cd build
cmake ..
make -j
```

编译成功会在`build`文件夹下生成`libtpuchat.so*`, 此时可以在web_demo.py中指定bmodel\_path token\_path device\_id, lib_path(编译生产的`libtpuchat.so*`文件, 默认路径是`./build`下), 以及dev_id。
```python
python3 web_demo.py
```
即可成功运行web的demo。
* PS：在用户不修改上述token\_path的lib\_path的存放路径前提下只需指定bmodel\_path即可运行程序。

如果是SoC环境，参考C++版本

* PS：尽量下载gradio==3.39.0版本，不然会出现各种问题！！

# 常见问题
* 请根据实际block数目调整`demo/chat`中或者`web_demo/chat.cpp`中的NUM_LAYERS，默认是使用Baichuan2-7B(NUM_LAYERS=32)
