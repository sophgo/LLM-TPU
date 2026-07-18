# Quick Start

If you have an SoC board or a PCIE board with the 1684X chip, such as Airbox (SoC), you can refer to the following steps to run a large language model. Here we take Qwen3-4B as an example.

In addition, the execution steps for SoC are somewhat different from those for PCIE. For PCIE, it is recommended to install docker before running. They are explained separately here.

## Run the Demo

### How to run the Demo on SoC

#### 1. Clone the LLM-TPU project and execute the run.sh script
```
git clone https://github.com/sophgo/LLM-TPU.git
./run.sh --model qwen3
```
If prompted that python libraries are missing, just pip3 install or pip3 install xxx --upgrade

### How to run the Demo on PCIE

#### 1. Install docker and enter docker
```
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name mlir -v /dev:/dev -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash
docker exec -it mlir bash
```

#### 2. Clone the LLM-TPU project and execute the run.sh script
```
git clone https://github.com/sophgo/LLM-TPU.git
./run.sh --model qwen3
```

## Version check

Before starting, first check whether the sophon-driver version meets the requirements

### How to perform the version check on SoC
```
uname -v
```
Afterwards, a time similar to `#2 SMP Sat Nov 18 10:07:36 HKT 2023` will be displayed. If your date >= 20240110, i.e. relatively new, then skip this step. If the date < 20240110, i.e. the version is relatively old, then refer to [this link](https://doc.sophgo.com/sdk-docs/v23.09.01-lts/docs_latest_release/docs/SophonSDK_doc/zh/html/sdk_intro/5_update.html#soc) to reinstall the sdk, and obtain the flash image package with the following command
```
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/sdcard.tgz
```

### How to perform the version check on PCIE
```
cat /proc/bmsophon/driver_version
```
Afterwards, a release date similar to `release version:0.5.1   release date: 20240717-103602` will be displayed. If your date >= 20240717, i.e. relatively new, then skip this step. If the date < 20240717, i.e. the version is relatively old, then reinstall the driver following the steps below
```
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/libsophon_club/20240717/sophon-driver_0.5.1_amd64.deb
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/libsophon_club/20240717/sophon-libsophon-dev_0.5.1_amd64.deb
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/libsophon_club/20240717/sophon-libsophon_0.5.1_amd64.deb

sudo apt remove sophon-driver sophon-libsophon
sudo dpkg -i sophon-*.deb
```
