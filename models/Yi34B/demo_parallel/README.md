# Yi-34B
Currently supports 2/4/8-chip int4/int8
## 1. Install the Driver
Download and install the driver with the following commands. **Note: this specific driver version is currently required; older driver versions do not support the latest multi-chip models**:
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/drivers/libsophon-0425deb.tar.gz
tar -xzf libsophon-0425deb.tar.gz
cd libsophon-0425deb
sudo apt remove sophon-driver sophon-libsophon
sudo dpkg -i *.deb

```

## 2. Compile
See ../README.md

## 3. Run
```shell
mkdir build && cd build
cmake .. && make -j8
./yi_parallel --model path_to_bmodel --devid 0,1 --tokenizer ../../support/token_config/tokenizer.model
```

## 4. FAQ

1) If the model loading process is interrupted suddenly for no apparent reason, you can run `ulimit -HSn 65536` to increase system resources.

2) If the model performance is much worse than expected or it hangs during runtime, run `test_cdma_p2p 0x130000000 0 0x140000000 1 0x100000` to test p2p performance. If the bandwidth is only around 1500MB/s, p2p may be unavailable. Enable it with the following steps:
    - iommu is not disabled. Disable it as follows:
      ```bash
      sudo vi /etc/default/grub
      # Add intel_iommu=off/amd_iommu=off according to the CPU type
      # GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet intel_iommu=off"
      # GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet amd_iommu=off"
      sudo update-grub
      sudo reboot
      ```
    - If the speed still does not improve after disabling iommu, you may also need to configure the PCIE link. Run the following commands, and then reinstall the driver:
      ```bash
      sudo setpci -v -s 99:*.0 ecap_acs+6.w=0
      sudo setpci -v -s 32:*.0 ecap_acs+6.w=0
      ```
