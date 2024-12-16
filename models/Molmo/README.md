# Llama3.2

本项目实现BM1684X部署语言大模型[Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

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
├── run_demo.sh                             #自动测试脚本
└── processor_config                        #分词器和预处理等配置
    ├── special_tokens_map.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ```
----------------------------

#  自动化推理脚本



# 【阶段一】模型编译

## 注意点
* 模型编译必须要在docker内完成，无法在docker外操作

### 步骤一：模型下载
可以通过huggingface或ModelScope官方下载
[huggingface](https://huggingface.co/allenai/Molmo-7B-D-0924)
[ModelScope](https://modelscope.cn/models/LLM-Research/Molmo-7B-D-0924/)


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
cp ./compile/files/Molmo-7B-D-0924/* $your_model_path
```

### 步骤五：生成onnx文件

``` shell
cd compile
python export_onnx.py -m $your_model_path -s 1024 -i 384
```

* PS1：-s为模型sequence长度，默认1024，不建议导出1024长度以下的模型，因为image token会占用seq len通常都在512长度以上
* PS2：-i为图像大小，默认384，表示输入图像为384x384，如果输入图像宽高不一致的图像可以手动修改脚本export_onnx.py 44行
* PS3：图像大小必须指定，目的是生成对应的size的fake input保存为权重，用于后续生成静态模型

### 步骤六：生成bmodel文件

生成单芯模型

``` shell
./compile.sh --mode int4 --name molmo-7b --seq_length 1024 # same as int8
```
* PS1：生成bmodel耗时大概3小时以上，建议64G内存以及200G以上硬盘空间，不然很可能OOM或者no space left
* PS2：--name必须指定为molmo-7b
* PS3：步骤三到步骤六可以通过运行compile文件夹下的run_compile.sh完成，具体命令是：
``` shell
./run_compile.sh --model_name molmo-7b --seq_length 1024 --model_path $your_model_path --tpu_mlir_path $your_mlir_path
```
如果没有填写model_path，脚本会从modelscope下载模型，如果没有填写mlir_path，脚本会通过dfss下载对应的tpu_mlir压缩包并解压
----------------------------

# 【阶段二】可执行文件生成

## 编译程序(Python Demo版本)
执行如下编译，(PCIE版本与SoC版本相同)：

```shell
cd python_demo
mkdir build && cd build
cmake .. && make
cp *chat* ..
```

## 模型推理(Python Demo版本)
```shell
cd ./python_demo
python3 pipeline.py -m bmodel_path -i image_path -s image_size -t ../processor_config --devid your_devid
```
注意：image size必须指定为export onnx时的image size，pipeline中会将输入图像resize到image size
其它可用参数可以通过`pipeline.py` 或者执行如下命令进行查看 
```shell
python3 pipeline.py --help
```
