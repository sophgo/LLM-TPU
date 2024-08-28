# 多芯demo

## 1. 安装驱动
按如下命令下载并安装驱动，**注意使用0611版本的驱动**：
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/drivers/libsophon-0611deb.tar.gz
tar -xzf libsophon-0611deb.tar.gz
cd libsophon-0611deb
sudo apt remove sophon-driver sophon-libsophon
sudo dpkg -i *.deb
```
驱动安装好后，便可以在直接执行`./run_demo_parallel.sh`来运行本demo。


## 2. 编译
编译模型需要在docker中进行，按如下命令进入docker中：
```shell
docker pull sophgo/tpuc_dev:latest
# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v /dev:/dev -v /opt/sophon:/opt/sophon -v /etc/profile.d:/etc/profile.d -v /etc/ld.so.conf.d:/etc/ld.so.conf.d -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```

按如下操作编译72b模型：
```shell
cd ../Qwen2/compile/
pushd files/Qwen2-72B-Instruct/
./replace_file.sh
popd
python3 ./export_onnx_parallel.py -m path_to/Qwen2-72B-Instruct/ --num_threads 72 --lmhead_with_topk 1
# 静态编译
./compile_onnx_parallel.sh --mode int4 --num_device 8 --name qwen2-72b --addr_mode io_alone --seq_length 8192 --dynamic 0
```

按如下操作编译7b模型：
```shell
cd ../Qwen2/compile/
pushd files/Qwen2-7B-Instruct/
./replace_file.sh
popd
python3 ./export_onnx_parallel.py -m path_to/Qwen2-7B-Instruct/ --num_threads 72 --lmhead_with_topk 1
# 静态编译
./compile_onnx_parallel.sh --mode int4 --num_device 8 --name qwen2-7b --addr_mode io_alone --seq_length 8192 --dynamic 0
```

如果不打算编译模型，可以通过以下命令下载已编译好的模型，目前有如下模型已经预编译好，**注意最新版本的驱动需要重新下载下方的模型**：
```shell
pip3 install dfss
# 72b 静态
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen2-72b_int4_seq8192_8dev_static.bmodel
# 7b 静态
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen2-7b_int4_seq8192_8dev_static.bmodel
```

## 3. 运行
```shell
git submodule update --init

cd python_demo_parallel
mkdir build 
cd build && cmake .. && make -j8 && cp *cpython* .. && cd ..
python3 pipeline.py --model_path ./qwen2-7b_int4_seq8192_8dev_static.bmodel --tokenizer_path ../support/token_config/ --devid 0,1,2,3,4,5,6,7
```

运行web demo
```shell
pip3 install gradio==3.39.0 mdtex2html==1.2.0 dfss
python3 web_demo.py --model_path ./qwen2-7b_int4_seq8192_8dev_static.bmodel --tokenizer_path ../support/token_config/ --devid 0,1,2,3,4,5,6,7
```

## 4. 常见问题

1) 模型加载过程中无缘无故突然中断，可以运行`ulimit -HSn 65536`增加系统资源

2) 模型性能比预期的差很多或者运行期间hang死，运行`test_cdma_p2p 0x130000000 0 0x140000000 1 0x100000`测试p2p性能，若带宽只有1500MB/s左右，可能是p2p不可用，按如下步骤开启：
    - iommu没有关闭，按如下过程关闭：
      ```bash
      sudo vi /etc/default/grub
      # 根据CPU类型选择添加intel_iommu=off/amd_iommu=off
      # GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet intel_iommu=off"
      # GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet amd_iommu=off"
      sudo update-grub
      sudo reboot
      ```
    - iommu关闭后速度依然上不来，可能还需要配置一下PCIE链路，运行如下命令，之后再重新安装驱动：
        - 运行如下命令，确定卡的编号：
        ```bash
        lspci | grep 4052
        # 如果只有一张卡，显示可能如下，82便是卡的编号。多张卡会显示多个
        # 81:00.0 PCI bridge: PMC-Sierra Inc. Device 4052
        # 82:00.0 PCI bridge: PMC-Sierra Inc. Device 4052
        # 82:01.0 PCI bridge: PMC-Sierra Inc. Device 4052
        # 82:02.0 PCI bridge: PMC-Sierra Inc. Device 4052
        # 82:03.0 PCI bridge: PMC-Sierra Inc. Device 4052
        # 82:04.0 PCI bridge: PMC-Sierra Inc. Device 4052
        # 82:05.0 PCI bridge: PMC-Sierra Inc. Device 4052
        # 82:06.0 PCI bridge: PMC-Sierra Inc. Device 4052
        # 82:07.0 PCI bridge: PMC-Sierra Inc. Device 4052
        ```
        - 配置PCIE链路，每张卡都需要运行如下命令：
        ```bash
        sudo setpci -v -s 82:*.0 ecap_acs+6.w=0
        ```
