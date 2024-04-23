![image](./assets/sophgo_chip.png)

# Llama3

本项目实现BM1684X部署语言大模型[Llama3-8B-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

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
│   └── pipeline.py                         #Llama3 python_demo的执行脚本
├── requirements.txt                        #环境配置所需安装的wheel包
├── run_demo.sh                             #自动测试脚本
└── token_config                            #分词器
    ├── special_tokens_map.json
    ├── tokenizer.json
    └── tokenizer_config.json
```
----------------------------

#  自动化推理脚本



# 【阶段一】模型编译

## 注意点
* 模型编译必须要在docker内完成，无法在docker外操作

### 步骤一：模型下载
虽然Llama3模型允许商业开源，但是模型下载需要想Meta提交使用申请，因此测试模型时可以参考[ModelScope提供的模型权重](https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct/summary)进行下载，或者通过Huggingface申请Meta License进行下载。


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
cp ./compile/files/Meta-Llama-3-8B-Instruct/modeling_llama.py /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
```
同时将`./compile/files/Meta-Llama-3-8B-Instruct/config.json` 替换下载好的`Llama-3-8B-Instruct`路径下的同名文件。

* PS：不一定是/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py这个路径，建议替换前先pip show transformers查看一下

### 步骤五：生成onnx文件

``` shell
cd compile
python export_onnx.py --model_path your_model_path --seq_length 512
```

* PS1：your_model_path 指的是原模型下载后的地址, 如:"../../Meta-Llama-3-8B-Instruct"。
* PS2：默认导出sequence length为512的模型

### 步骤六：生成bmodel文件

生成单芯模型

``` shell
./compile.sh --mode int8 --name llama3-8b --seq_length 512 # same as int4
```

生成双芯模型

``` shell
./compile.sh --mode int8 --num_device 2 --name llama3-8b --seq_length 512 # same as int4
```

* PS1：生成bmodel耗时大概3小时以上，建议64G内存以及200G以上硬盘空间，不然很可能OOM或者no space left
* PS2：如果想要编译llama3-8b，则--name必须为llama3-8b
* PS3：目前给定的lib_pcie和lib_soc部分仅包含单芯的动态库，多芯部分会在后续更新

----------------------------

# 阶段二：可执行文件生成

## 编译程序(Python Demo版本)
执行如下编译，(PCIE版本与SoC版本相同)：

```shell
cd python_demo
mkdir build
cd build
cmake ..
make
cp *chat* ..
```

## 模型推理(Python Demo版本)
```shell
cd ./python_demo
python3 pipeline.py -m your_model_path -t ../token_config --devid your_devid
```
其它可用参数可以通过`pipeline.py` 或者执行如下命令进行查看 
```shell
python3 pipeline.py --help
```

