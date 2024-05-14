# Yi-34B
目前支持2/4/8芯int4/int8
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

## 2. 编译
见../README.md

## 3. 运行
```shell
mkdir build && cd build
cmake .. && make -j8
./yi_parallel --model path_to_bmodel --devid 0,1 --tokenizer ../../support/tokenizer.model
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
      ```bash
      sudo setpci -v -s 99:*.0 ecap_acs+6.w=0
      sudo setpci -v -s 32:*.0 ecap_acs+6.w=0
      ```

