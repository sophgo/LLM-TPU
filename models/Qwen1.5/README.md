# Qwen1.5

## 目录
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 运行环境准备](#3-运行环境准备)
  - [4. 模型编译](#4-模型编译)
  - [5. 模型推理](#5-模型推理)

## 1. 简介
Qwen1.5 是Qwen的第二代版本，它是开源中英双语对话模型，关于它的特性，请前往源repo查看：https://huggingface.co/Qwen。本例程对Qwen进行移植，使之能在SOPHON BM1684X上进行推理测试。

该例程支持在V23.07.01(libsophon_0.4.9)及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行，也可以在1684X SoC设备（如SE7、SM7、Airbox等）上运行。在SoC上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。建议后续所有流程都在[提供的Docker](#32-docker-版本)中完成。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)
* 支持FP16、INT8、INT4模型编译和推理
* 支持基于pybing推理的Python例程

## 3. 运行环境准备
在PCIe上无需修改内存，以下为soc模式相关：
对于1684X系列设备（如SE7/SM7），都可以通过这种方式完成环境准备，使得满足Qwen运行条件。首先，在1684x SoC环境上，参考如下命令修改设备内存。
```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/09/11/13/DeviceMemoryModificationKit.tgz
tar xvf DeviceMemoryModificationKit.tgz
cd DeviceMemoryModificationKit
tar xvf memory_edit_{vx.x}.tar.xz #vx.x是版本号
cd memory_edit
./memory_edit.sh -p #这个命令会打印当前的内存布局信息
./memory_edit.sh -c -npu 7615 -vpu 3072 -vpp 3072 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/boot.itb /boot/boot.itb && sync
sudo reboot
```
> **注意：**
> 1. tpu总内存为npu/vpu/vpp三者之和，fp16模型应满足tpu内存 >= 12800 MB，int8应满足tpu内存 >= 7168MB，int4应满足tpu内存 >= 4608MB。
> 2. 更多教程请参考[SoC内存修改工具](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html)

### 3.1 mlir版本
  从sftp上获取TPU-MLIR压缩包
  ```bash
  pip3 install dfss --upgrade
  python3 -m dfss --url=open@sophgo.com:sophon-demo/Qwen/tpu-mlir_v1.6.113-g7dc59c81-20240105.tar.gz 
  tar -xf tpu-mlir_v1.6.113-g7dc59c81-20240105.tar.gz 
  ```

### 3.2 docker 版本
  TPU-MLIR使用的docker是sophgo/tpuc_dev:latest, docker镜像和tpu-mlir有绑定关系，少数情况下有可能更新了tpu-mlir，需要新的镜像。
  ```bash
  docker pull sophgo/tpuc_dev:latest
  # 这里将本级目录映射到docker内的/workspace目录,用户需要根据实际情况将demo的目录映射到docker里面
  # myname只是举个名字的例子, 请指定成自己想要的容器的名字
  docker run --name myname -v $PWD:/workspace -it sophgo/tpuc_dev:latest
  # 此时已经进入docker，并在/workspace目录下
  # 初始化软件环境
  cd /workspace/tpu-mlir_vx.y.z-<hash>-<date>
  source ./envsetup.sh
  ```
其中`tpu-mlir_vx.y.z-<hash>-<date>`的路径请使用[下载得到的mlir路径](#31-mlir版本)，更多TPU-MLIR的教程请参考[算能官网](https://developer.sophgo.com/site/index/material/31/all.html)的《TPU-MLIR快速入门手册》和《TPU-MLIR开发参考手册》。

## 4. 模型编译
## 4.1 获取onnx
### 4.1.1 下载Qwen1.5官方代码

**注：** Qwen1.5-7B官方库50G左右，在下载之前，要确认自己有huggingface官网的access token或者SSH key。（Qwen1.5-1.8B / Qwen1.5-14B的操作相同，请保证满足对应内存需求）,以下代码以Qwen1.5-7B为例

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen1.5-7B-Chat
```
如果git clone完代码之后出现卡住，可以尝试`ctrl+c`中断，然后进入仓库运行`git lfs pull`。

### 4.1.2 修改官方代码：
本例程的`tools`目录下提供了修改好之后的`config.json`和`modeling_qwen2.py`。(transformers请更新到4.38.2以上)可以直接替换掉原仓库的文件：

```bash
cp compile/Qwen1.5-1.8B-Chat/config.json Qwen1.5-1.8B-Chat/
cp compile/Qwen1.5-1.8B-Chat/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/
```

### 4.1.3 导出onnx
- 导出所有onnx模型，如果过程中提示缺少某些组件，直接**pip install**组件即可
- (<strong>其中torch请勿使用2.1.1及以上版本，否则会出现SPDA相关报错，建议torch==2.0.1,transformers==4.38.2</strong>)

```bash
# 将/workspace/Qwen-7B-Chat换成docker环境中您的Qwen-7B-Chat仓库的路径
python3 compile/export_onnx.py --model_path /workspace/Qwen1.5-1.8B-Chat --sequence_length your_sequence_length
```
此时有大量onnx模型被导出到本例程中`compile/tmp/onnx`的目录。

### 4.2 bmodel编译
首先需要在mlir工具下激活环境，如果之前没有下载mlir工具，[mlir下载地址可参考](./Qwen_Export_Guide.md/#212-下载并解压tpu-mlir)
```bash
cd tpu-mlir_v1.6.113-g7dc59c81-20240105
source envsetup.sh
```
目前TPU-MLIR支持1684x对Qwen1.5进行BF16(仅限Qwen-1.8B),INT8和INT4量化，使用如下命令生成bmodel。

```bash
./compile.sh --mode int4 --name qwen1.5-1.8b
```

其中，mode可以指定bf16/int8/int4，编译成功之后，模型将会存放在`compile`目录下。

### 4.3 准备tokenizer
tokenizer已经放在了`support`文件夹下

## 5. 模型推理
您可以使用我们已经编译好的模型进行自动化编译，也可以使用自己编译的模型进行推理
### 5.1 自动化测试
在Qwen1.5路径下执行（默认测试模型为Qwen1.5-1.8B）
```bash
./run_demo.sh
```
### 5.2 测试自己的模型
在python_demo中编译模型推理动态库
```bash
cd python_demo
mkdir build && cd build
cmake ..
make -j4
cp chat.cpython-310-x86_64-linux-gnu.so ..
cd ..
```

此时可以进行模型推理
```bash
source ../../../envsetup.sh # 模型推理前请确保在LLM-TPU的路径下激活环境变量`source envsetup.sh`.
python chat.py --devid '0' --model_path your_bmodel_path --tokenizer_path ../support/token_config/
```
其中`your_bmodel_path`请以实际路径为准

