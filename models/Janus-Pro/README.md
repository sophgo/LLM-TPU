# Janus-Pro-7b

本项目实现BM1684X部署语言大模型[Deepseek-Janus-Pro-7b](https://huggingface.co/deepseek-ai/Janus-Pro-7B)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

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
│   └── pipeline.py                         #python_demo的执行脚本
├── requirements.txt                        #环境配置所需安装的wheel包
└── processor_config                        #分词器和预处理等配置
    ├── special_tokens_map.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ...
```
----------------------------

#  自动化推理脚本

# 【阶段一】模型编译

## 注意点
* 模型编译必须要在docker内完成，无法在docker外操作, 如果不打算编译模型，也可以使用我们编译好的模型，直接跳转至[编译程序](##-编译程序)

### 步骤一：模型下载
可以通过huggingface或ModelScope官方下载
[huggingface](https://huggingface.co/deepseek-ai/Janus-Pro-7B)


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
```

### 步骤五：生成bmodel文件

生成单芯模型

``` shell
llm_convert.py -m /path/to/Janus-Pro-1B/ -s 710 -q bf16 -g 128 --num_device 1  -c bm1684x  -o janus/ # same as int8
```

----------------------------

# 【阶段二】可执行文件生成

## 编译程序
如果不打算自己编译模型，可以直接用下载好的模型
```
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/janus-pro-7b_int4_seq2048.bmodel
```
```
python3 -m dfss --url=open@sophgo.com:/share/hengyang/janus-pro-1b_bf16_seq710_bm1684x_1dev_20251021_154608.bmodel(1b模型)
```

执行如下编译，(PCIE版本与SoC版本相同)：

```shell
cd python_demo
mkdir build && cd build
cmake .. && make
cp *chat* ..
```

## 模型推理
```shell
cd ./python_demo
python3 pipeline.py -m bmodel_path -i image_path -t ../support/processor_config --devid your_devid
```

* 其它可用参数可以通过`pipeline.py` 或者执行如下命令进行查看 
```shell
python3 pipeline.py --help
```
