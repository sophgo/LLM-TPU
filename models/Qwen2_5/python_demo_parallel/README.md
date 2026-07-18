# Multi-chip demo

```
pip3 install dfss
# 2 chips
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-14b_int4_seq8192_2dev.bmodel
# 4 chips
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-14b_int4_seq8192_4dev.bmodel
# 8 chips
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-14b_int4_seq8192_8dev.bmodel
```

## Run
```shell
mkdir build
cd build && cmake .. && make -j8 && cp *cpython* .. && cd ..
python3 pipeline.py --model_path ./qwen2.5-14b_int4_seq8192_8dev.bmodel --tokenizer_path ../support/token_config/ --devid 0,1,2,3,4,5,6,7
```

## FAQ

1) If the model loading process suddenly interrupts for no apparent reason, you can run `ulimit -HSn 65536` to increase system resources

2) If the model performance is much worse than expected or it hangs during running, run `test_cdma_p2p 0x130000000 0 0x140000000 1 0x100000` to test p2p performance. If the bandwidth is only around 1500MB/s, p2p may be unavailable. Enable it with the following steps:
    - iommu is not disabled. Disable it with the following process:
      ```bash
      sudo vi /etc/default/grub
      # Add intel_iommu=off/amd_iommu=off according to the CPU type
      # GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet intel_iommu=off"
      # GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet amd_iommu=off"
      sudo update-grub
      sudo reboot
      ```
    - If the speed still does not increase after disabling iommu, you may also need to configure the PCIE link. Run the following commands, and then reinstall the driver:
        - Run the following command to determine the card number:
        ```bash
        lspci | grep 4052
        # If there is only one card, the output may look like the following, where 82 is the card number. Multiple cards will show multiple entries
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
        - Configure the PCIE link. The following command needs to be run for each card:
        ```bash
        sudo setpci -v -s 82:*.0 ecap_acs+6.w=0
        ```
