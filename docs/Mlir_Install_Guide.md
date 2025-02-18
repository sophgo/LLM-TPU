# MLIR环境安装指南

## 目录
- [MLIR环境安装指南](#MLIR环境安装指南)
  - [目录](#目录)
  - [1. MLIR简介](#1-MLIR简介)
  - [2. 配置docker](#2-配置docker)
    - [2.1 docker安装](#21-docker安装)
    - [2.2 docker创建](#22-docker创建)
  - [3. 配置MLIR](#3-配置MLIR)
    - [3.1 从pypi下载安装MLIR](#31-从pypi下载安装MLIR)
    - [3.2 从github下载MLIR包](#32-从github下载MLIR包)
    - [3.2 从release包直接安装](#33-从release包直接安装)


## 1 MLIR简介
TPU-MLIR是算能深度学习处理器的编译器工程。该工程提供了一套完整的工具链, 可以将不同框架下预训练的神经网络, 转化为可以在算能智能视觉深度学习处理器上高效运算的文件bmodel。代码已经开源到[github](https://github.com/sophgo/tpu-mlir)


使用TPU-MLIR编译BModel，通常需要在x86主机上安装TPU-MLIR环境，x86主机已安装Ubuntu16.04/18.04/20.04系统，并且运行内存在12GB以上。


建议TPU-MLIR的镜像仅用于编译和量化模型，程序编译和运行请在开发和运行环境中进行。更多TPU-MLIR的教程请参考[算能官网](https://developer.sophgo.com/site/index.html?categoryActive=material)的《TPU-MLIR快速入门手册》和《TPU-MLIR开发参考手册》。

## 2 配置docker

### 2.1 docker安装
若已安装docker，请跳过本节。
    
```bash
# 如果您的docker环境损坏，可以先卸载docker
sudo apt-get remove docker docker.io containerd runc

# 安装依赖
sudo apt-get update
sudo apt-get install \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

# 安装 docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# docker命令免root权限执行
# 创建docker用户组，若已有docker组会报错，没关系可忽略
sudo groupadd docker
# 将当前用户加入docker组
sudo usermod -aG docker $USER
# 切换当前会话到新group或重新登录重启X会话
newgrp docker​ 
```
> **提示**：需要logout系统然后重新登录，再使用docker就不需要sudo了。

### 2.2 docker创建

TPU-MLIR使用的docker是sophgo/tpuc_dev:latest, docker镜像和tpu-mlir有绑定关系，少数情况下有可能更新了tpu-mlir，需要新的镜像。
```bash
docker pull sophgo/tpuc_dev:latest
# 这里将本级目录映射到docker内的/workspace目录,用户需要根据实际情况将demo的目录映射到docker里面
# myname只是举个名字的例子, 请指定成自己想要的容器的名字
docker run --privileged --name myname --network host -v $PWD:/workspace -it sophgo/tpuc_dev:latest
# 此时已经进入docker，并在/workspace目录下  
```

## 3 配置MLIR
MLIR目前支持三种安装方法：

### 3.1 从pypi下载安装MLIR
直接从pypi下载并安装：
```bash
pip install tpu_mlir -i https://pypi.tuna.tsinghua.edu.cn/simple 
```

### 3.2 从github下载MLIR包
从[TPU-MLIR Github](https://github.com/sophgo/tpu-mlir/releases)下载最新`tpu_mlir-*-py3-none-any.whl`，然后使用pip安装：
```bash
pip install tpu_mlir-*-py3-none-any.whl
```

TPU-MLIR在对不同框架模型处理时所需的依赖不同，对于onnx或torch生成的模型文件，
使用下面命令安装额外的依赖环境:
```bash
pip install tpu_mlir[onnx]
pip install tpu_mlir[torch]
```
目前支持五种配置: onnx, torch, tensorflow, caffe, paddle。可使用一条命令安装多个配置，也可直接安装全部依赖环境:
```bash
pip install tpu_mlir[onnx,torch,caffe]
pip install tpu_mlir[all]
```

### 3.3 从release包直接安装
如果您获取了类似`tpu-mlir_${version}-${hash}-${date}.tar.gz`这种形式的发布包，可以通过这种方式配置：
```bash
# 如果此前有通过pip安装过mlir，需要卸载掉
pip uninstall tpu_mlir

tar xvf tpu-mlir_${version}-${hash}-${date}.tar.gz
cd tpu-mlir_${version}-${hash}-${date}
source envsetup.sh #配置环境变量
```
