# 多芯demo

## 1. 编译
编译模型需要在docker中进行，按如下命令进入docker中：
```shell
docker pull sophgo/tpuc_dev:latest
# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v /dev:/dev -v /opt/sophon:/opt/sophon -v /etc/profile.d:/etc/profile.d -v /etc/ld.so.conf.d:/etc/ld.so.conf.d -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```

按如下操作编译qwen2.5-14b模型：
```shell
cd Qwen2_5/compile/
cp files/Qwen2.5-14B-Instruct/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/
# 多芯模型需要打开 lmhead_with_topk
python3 export_onnx.py -m path_to/Qwen2.5-14B-Instruct/ --seq_length 8192 --lmhead_with_topk 1
# 静态编译
./compile.sh --mode int4 --num_device 4 --name qwen2.5-14b --addr_mode io_alone --seq_length 8192 --dynamic 0
```

如果不打算编译模型，可以通过以下命令下载已编译好的模型：
```shell
pip3 install dfss
# 两芯
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-14b_int4_seq8192_2dev.bmodel
# 四芯
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-14b_int4_seq8192_4dev.bmodel
# 八芯
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-14b_int4_seq8192_8dev.bmodel
```

## 2. 运行
```shell
git submodule update --init

cd ../python_demo_parallel
mkdir build
cd build && cmake .. && make -j8 && cp *cpython* .. && cd ..
python3 pipeline.py --model_path ./qwen2.5-14b_int4_seq8192_8dev.bmodel --tokenizer_path ../support/token_config/ --devid 0,1,2,3,4,5,6,7
```

## 3. 常见问题

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
