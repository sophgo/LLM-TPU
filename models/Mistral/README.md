![](./assets/tpumlir.png)

# Qwen

本工程实现BM1684X部署语言大模型[Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。


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

下载本项目后（可通过下载：
``` shell
git clone git@github.com:sophgo/LLM-TPU.git
```
更新第三方库:
``` shell
cd LLM-TPU
git submodule update --init
```

### 4. 下载pytorch.bin模型

``` shell
cd LLM-TPU/models/Mistral/
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
cp compile/files/Mistral-7B-Instruct-v0.2/* Mistral-7B-Instruct-v0.2/
export PYTHONPATH=$PWD/Mistral-7B-Instruct-v0.2:$PYTHONPATH

cd compile
python3 export_onnx.py --model_path ../Mistral-7B-Instruct-v0.2
```

该工程比较大，会花较长时间。
在导出onnx前，请确保`files/Mistral-7B-Instruct-v0.2`中的文件已经替换了`Mistral-7B-Instruct-v0.2`下的对应文件。（默认sequence length为512）

## 5. 编译模型

注意此时在Docker环境workspace目录。

目前TPU-MLIR支持对`Mistral-7B-Instruct-v0.2`进行FP16、INT8和INT4量化，且支持多芯分布式推理，默认情况下会进行INT8量化和单芯推理，最终生成`mistral-7b_int4_1dev.bmodel`文件。（请先确保之前执行了[mlir的编译与环境的激活](#2-下载tpu-mlir代码并编译)).

```shell
cd LLM-TPU/models/Mistral-7B-Instruct-v0.2/compile
./compile.sh --name mistral-7b --mode int4 --addr_mode io_alone # int4 (defaulted)
```

若想进行2芯推理，则执行以下命令，最终生成`mistral-7b_int4_2dev.bmodel`文件，4芯8芯同理：

```shell
./compile.sh --num_device 2 --name mistral-7b --mode int4 --addr_mode io_alone
```

## 6. 使用Sophgo提供的模型
您也可以使用Sophgo已经编译好的模型进行后续推理，其使用方式为
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/mistral-7b_int4_1dev.bmodel
```

## 7. 编译程序(Python)

执行如下编译 (注意如果是SoC版本，需要把demo目录拷贝到SoC环境编译)：

```shell
cd /workspace/LLM-TPU/models/Mistral/python_demo
mkdir build
cd build
cmake ..
make
cp chat.cpython-310-x86_64-linux-gnu.so ..
cd ..
```
如果提示缺少`pybind`的文件，请通过下列命令
```shell
sudo apt-get update
sudo apt-get install pybind11-dev
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

