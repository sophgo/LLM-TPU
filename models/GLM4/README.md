![image](./assets/sophgo_chip.png)

# ChatGLM4

本项目实现BM1684X部署语言大模型[glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

下文中默认是PCIE环境；如果是SoC环境，按提示操作即可。

# 目录说明
```
.
├── README.md
├── compile
│   ├── compile.sh                          #用来编译TPU模型的脚本
│   ├── export_onnx.py                      #用来导出onnx的脚本
│   └── files                               #用于替换原模型的文件
├── python_demo
│   ├── chat.cpp                            #主程序文件
│   ├── pipeline.py                         #ChatGLM4 python demo的执行脚本
│   └── web_demo.py                         #ChatGLM4 web demo的执行脚本
├── requirements.txt                        #环境配置所需安装的wheel包
├── run_demo.sh                             #自动测试脚本
└── token_config                            #分词器
    ├── tokenization_chatglm.py
    ├── tokenizer_config.json
    └── tokenizer.model
```
----------------------------

#  自动化推理脚本



# 【阶段一】模型编译

如果不打算自己编译模型，可以直接下载编译好的模型：
```bash
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/glm4-9b_int4_1dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/glm4-9b_int8_1dev.bmodel
```
## 注意点
* 模型编译必须要在docker内完成，无法在docker外操作。

### 步骤一：下载docker

下载docker，启动容器，如下：

```bash
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
* PS：本repo `LLM-TPU`需在当前目录内

### 步骤二：下载TPU-MLIR代码并安装

``` shell
pip3 install dfss  --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/tpu-mlir.tar.gz
tar zxvf tpu-mlir.tar.gz
cd tpu-mlir_v1.8.beta.0-134-g859a6f517-20240801
source ./envsetup.sh
cd ..
```
* PS：重新进入docker环境并且需要编译模型时，必须在此路径下执行上述`source ./envsetup.sh`才能完成后续模型编译。

### 步骤三：模型下载
ChatGLM4模型允许商业开源，可以通过Huggingface官网下载[glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat)。
如果无法从官网下载，这里也提供一个下载好的压缩包。
```bash
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/glm-4-9b-chat-torch.zip
unzip glm-4-9b-chat-torch.zip
```

下载完`glm-4-9b-chat`官方库后，您还需要设置`ChatGLM4_PATH`环境变量，模型导出时会使用到。
```bash
export ChatGLM4_PATH=$PWD/glm-4-9b-chat
```

### 步骤四：对齐模型环境

```bash
sudo apt-get update
sudo apt-get install pybind11-dev
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cp ./compile/files/glm-4-9b-chat/modeling_chatglm.py $ChatGLM4_PATH
cp ./compile/files/glm-4-9b-chat/config.json $ChatGLM4_PATH
```

### 步骤五：生成onnx文件

```bash
cd compile
python export_onnx.py --model_path $ChatGLM4_PATH --seq_length 512
```
* PS：默认导出sequence length为512的模型。导出其它长度的模型，还需同步修改`$ChatGLM4_PATH/config.json`中的`seq_length`参数。

### 步骤六：生成bmodel文件

生成单芯模型

```bash
./compile.sh --mode int4 --name glm4-9b --seq_length 512 --addr_mode io_alone
```
生成W8A16量化的模型
```bash
./compile.sh --mode int8 --name glm4-9b --seq_length 512 --addr_mode io_alone
```
生成8192长度的模型
```bash
./compile.sh --mode int8 --name glm4-9b --seq_length 8192 --addr_mode io_alone
```


<!-- 生成双芯模型

```bash
./compile.sh --mode int4 --num_device 2 --name glm4-9b --seq_length 512 # same as int8
``` -->

* PS1：生成bmodel耗时大概3小时以上，建议64G内存以及200G以上硬盘空间，不然很可能OOM或者no space left；
* PS2：如果想要编译glm4-9b，则--name必须为glm4-9b。
<!-- * PS3：目前给定的lib_pcie和lib_soc部分仅包含单芯的动态库，多芯部分会在后续更新。 -->

----------------------------

# 阶段二：可执行文件生成

## 编译程序(Python Demo版本)
执行如下编译，(PCIE版本与SoC版本相同)：

```bash
cd python_demo
mkdir build && cd build
cmake ..
make
cp *chat* ..
```

## 模型推理(Python Demo版本)
```bash
cd ./python_demo
python3 pipeline.py --model_path glm4-9b_int4_1dev.bmodel --tokenizer_path ../token_config --devid your_devid
```
其它可用参数可以通过`pipeline.py`或者执行如下命令进行查看。
```bash
python3 pipeline.py --help
```

## web demo
```bash
python3 web_demo.py --model_path glm4-9b_int4_1dev.bmodel --tokenizer_path ../token_config --devid 0
```
