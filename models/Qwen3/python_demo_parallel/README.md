# 多芯demo

## 模型编译

```shell
# --num_device 2 指定2芯
llm_convert.py -m /workspace/Qwen3-14B-AWQ  -s 2048 --quantize w4bf16  -c bm1684x --out_dir qwen3_2dev --num_device 2
```

已编译模型：
```shell
pip3 install dfss
# 2芯
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-14b-awq_w4bf16_seq2048_bm1684x_2dev_20250812_144515.bmodel
# 4芯
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-14b-awq_w4bf16_seq2048_bm1684x_4dev_20250812_145320.bmodel
```

## 运行
```shell
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
python3 pipeline.py -m ./qwen3-xxx-4dev.bmodel -c ../config/ --devid 0,1,2,3
```

## 常见问题

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
