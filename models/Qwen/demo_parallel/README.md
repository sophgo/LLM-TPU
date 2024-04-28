# 多芯demo

## 1. 安装驱动
按如下命令下载并安装驱动，**注意目前必须要这一版本的驱动，旧版本驱动不支持最新的多芯模型**：
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/drivers/libsophon-0425deb.tar.gz
tar -xzf libsophon-0425deb.tar.gz
cd libsophon-0425deb
sudo apt remove sophon-driver sophon-libsophon
sudo dpkg -i *.deb
```
驱动安装好后，便可以在直接执行`./run_demo_parallel.sh`来运行本demo。


## 2. 编译
编译模型需要在docker中进行，按如下命令进入docker中：
```shell
docker pull sophgo/tpuc_dev:latest
# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```

按如下操作编译模型：
```shell
cd Qwen/compile
python3 ./export_onnx.py -m path_to/Qwen-72B-Chat/ --num_threads 72 --lmhead_with_topk 1
./compile.sh --mode int4 --num_device 8 --addr_mode io_alone --seq_length 8192
```

如果不打算编译模型，可以通过以下命令下载已编译好的模型，目前有如下模型已经预编译好，**注意最新版本的驱动需要重新下载下方的模型**：
```shell
pip3 install dfss
# int4
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen-72b_int4_8192_8dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen-14b_int4_8dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen-14b_int4_6dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen-14b_int4_4dev.bmodel
# int8
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen-14b_int8_8dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen-14b_int8_6dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen-14b_int8_4dev.bmodel
```

## 3. 运行
```shell
cd demo
mkdir build && cd build
cmake .. && make -j8
./qwen --model path_to/qwen-72b_int4_8192_8dev.bmodel --devid 0,1,2,3,4,5,6,7 --tokenizer path_to/qwen.tiktoken
```

## 4. 常见问题

1) 模型加载过程中无缘无故突然中断，可以运行`ulimit -HSn 65536`增加系统资源

2) 模型性能比预期的差很多，运行`test_cdma_p2p 0x130000000 0 0x140000000 1 0x100000`测试p2p性能，若带宽只有1500MB/s左右，可能是iommu没有关闭，按如下过程关闭：
```bash
sudo vi /etc/default/grub
# 根据CPU类型选择添加intel_iommu=off/amd_iommu=off
# GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet intel_iommu=off"
# GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet amd_iommu=off"
sudo update-grub
sudo reboot
```
