![](./assets/tpumlir.png)

# Mistral

本工程实现BM1684X部署语言大模型[Yi-6B-Chat](https://huggingface.co/01-ai/Yi-6B-Chat)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。


## 开发环境准备

### 1. 下载docker，启动容器

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```
后文假定环境都在docker的`/workspace`目录。

### 2. 下载`TPU-MLIR`代码并编译

(也可以直接下载编译好的release包解压)

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  #激活环境变量
./build.sh #编译mlir
```

### 3. 更新第三方库

下载本项目：
``` shell
git clone git@github.com:sophgo/LLM-TPU.git
```
第三方库环境要求
``` shell
pip3 install transformers==4.39.1
pip3 install torch==2.0.1

sudo apt-get update
sudo apt-get install pybind11-dev
```
```

### 4. 下载pytorch.bin模型

``` shell
cd LLM-TPU/models/Mistral/
git lfs install
git clone https://huggingface.co/01-ai/Yi-6B-Chat
cp compile/files/Yi-6B-Chat/config.json Yi-6B-Chat
cp compile/files/Yi-6B-Chat/modeling_llama.py /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
export PYTHONPATH=$PWD/Yi-6B-Chat:$PYTHONPATH

cd compile
python3 export_onnx.py --model_path ../Yi-6B-Chat
```

该工程比较大，会花较长时间。
在导出onnx前，请确保`files/Yi-6B-Chat`中的文件已经替换了运行时实际使用的`transformers`下的对应文件。（默认sequence length为512）

## 5. 编译模型

注意此时在Docker环境workspace目录。

目前TPU-MLIR支持对`Yi-6B-Chat`进行FP16、INT8和INT4量化，且支持多芯分布式推理，默认情况下会进行INT8量化和单芯推理，最终生成`yi-6b_int4_1dev.bmodel`文件。（请先确保之前执行了[mlir的编译与环境的激活](#2-下载tpu-mlir代码并编译)).

```shell
cd LLM-TPU/models/Yi-6B-Chat/compile
./compile.sh --name yi-6b --mode int4 --addr_mode io_alone # int4 (defaulted)
```

若想进行2芯推理，则执行以下命令，最终生成`yi-6b_int4_1dev.bmodel`文件，4芯8芯同理：

```shell
./compile.sh --num_device 2 --name yi-6b --mode int4 --addr_mode io_alone
```

## 6. 使用Sophgo提供的模型
您也可以使用Sophgo已经编译好的模型进行后续推理，其使用方式为
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/yi-6b_int4_1dev.bmodel
```

## 7. 编译程序(Python)

执行如下编译 (注意如果是SoC版本，需要把demo目录拷贝到SoC环境编译)：

```shell
cd /workspace/LLM-TPU/models/Yi/python_demo
mkdir build
cd build
cmake ..
make
cp chat.cpython-310-x86_64-linux-gnu.so ..
cd ..
```

### a. 命令行交互
- 单芯推理：使用如下命令。
```shell
python3 pipeline.py --model your_bmodel_path --devid 0 # devid 默认使用 0 号进行推理
```
请根据实际bmodel的路径设置`your_bmodel_path`, 其它更多的参数可以通过
```shell
python3 pipeline.py --help
```
进行查看

