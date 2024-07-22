# 快速开始

如果您手上有1684X芯片的SoC板卡或者PCIE板卡，例如Airbox（SoC），那么可以参考以下步骤跑通大语言模型，这里以Llama2-7B为例。

另外SoC的执行步骤和PCIE的有些区别，PCIE必须要安装docker后才能运行，这里将其分开说明。

## 跑通Demo

### SoC如何跑通Demo

#### 1. 克隆LLM-TPU项目，并执行run.sh脚本
```
git clone https://github.com/sophgo/LLM-TPU.git
./run.sh --model llama2-7b
```

### PCIE如何跑通Demo

#### 1. 安装docker，并进入docker
```
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name mlir -v /dev:/dev -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash
docker exec -it mlir bash
```

#### 2. 克隆LLM-TPU项目，并执行run.sh脚本
```
git clone https://github.com/sophgo/LLM-TPU.git
./run.sh --model llama2-7b
```

## 版本检查

在开始之前，首先要检查sophon-driver的版本是否符合条件

### SoC如何执行版本检查
```
uname -v
```
之后，会显示类似这样的一个时间`#2 SMP Sat Nov 18 10:07:36 HKT 2023`，如果你的日期>=20240110，也就是比较新，那么跳过这一步，如果日期<20240110，也就是版本比较老，那么参考[这个链接](https://doc.sophgo.com/sdk-docs/v23.09.01-lts/docs_latest_release/docs/SophonSDK_doc/zh/html/sdk_intro/5_update.html#soc)重新安装sdk，刷机包则用以下命令获取
```
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/sdcard.tgz
```

### PCIE如何执行版本检查
```
cat /proc/bmsophon/driver_version
```
之后，会显示类似这样的一个release date`release version:0.5.0   release date: 20240304-175610`，如果你的日期>=20240110，也就是比较新，那么跳过这一步，如果日期<20240110，也就是版本比较老，那么按照如下步骤重新安装driver
```
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/sophon-driver_0.5.0_amd64.deb
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/sophon-libsophon-dev_0.5.0_amd64.deb
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/sophon-libsophon_0.5.0_amd64.deb

sudo apt remove sophon-driver sophon-libsophon
sudo dpkg -i sophon-*.deb
```
