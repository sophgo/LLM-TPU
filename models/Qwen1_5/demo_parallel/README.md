# Multi-chip demo

## 1. Install the driver
Download and install the driver with the following commands. **Note: use the 0611 driver version; older drivers have issues running dynamic Qwen1.5-32B**:
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/drivers/libsophon-0611deb.tar.gz
tar -xzf libsophon-0611deb.tar.gz
cd libsophon-0611deb
sudo apt remove sophon-driver sophon-libsophon
sudo dpkg -i *.deb
```
Once the driver is installed, you can directly run `./run_demo_parallel.sh` to run this demo.


## 2. Compile
Model compilation must be done inside docker. Enter docker with the following commands:
```shell
docker pull sophgo/tpuc_dev:latest
# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v /dev:/dev -v /opt/sophon:/opt/sophon -v /etc/profile.d:/etc/profile.d -v /etc/ld.so.conf.d:/etc/ld.so.conf.d -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```

Compile the model as follows:
```shell
cd Qwen/compile/
pushd files/Qwen1.5-32B-Chat/
./replace_file.sh
popd
python3 ./export_onnx.py -m path_to/Qwen1.5-32B-Chat/ --num_threads 72 --lmhead_with_topk 1
# Static compilation
./compile.sh --mode int4 --num_device 8 --addr_mode io_alone --seq_length 8192
# Dynamic compilation; note that the 0611 driver version is required
./compile.sh --mode int4 --num_device 8 --addr_mode io_alone --seq_length 8192 --dynamic 1
```

If you do not plan to compile the model, you can download a precompiled model with the following commands. The following models are currently precompiled. **Note: with the latest driver version, you need to re-download the models below**:
```shell
pip3 install dfss
# Static
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen1.5-32b_int4_seq8192_8dev_static.bmodel
# Dynamic
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen1.5-32b_int4_seq8192_8dev_dyn.bmodel
```

## 3. Run
```shell
git submodule update --init

cd demo_parallel
./link_third_party.sh
mkdir build && cd build
cmake .. && make -j8
./qwen --model path_to/qwen1.5-32b_int4_seq8192_8dev_dyn.bmodel --devid 0,1,2,3,4,5,6,7 --tokenizer path_to/qwen.tiktoken
```

## 4. FAQ

1) If the model loading process suddenly interrupts for no apparent reason, run `ulimit -HSn 65536` to increase system resources.

2) If the model performance is much worse than expected or it hangs during runtime, run `test_cdma_p2p 0x130000000 0 0x140000000 1 0x100000` to test p2p performance. If the bandwidth is only around 1500MB/s, p2p may be unavailable. Enable it as follows:
    - iommu is not disabled. Disable it as follows:
      ```bash
      sudo vi /etc/default/grub
      # Add intel_iommu=off or amd_iommu=off depending on your CPU type
      # GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet intel_iommu=off"
      # GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet amd_iommu=off"
      sudo update-grub
      sudo reboot
      ```
    - If the speed is still low after disabling iommu, you may also need to configure the PCIE link. Run the following commands, then reinstall the driver:
        - Run the following command to determine the card number:
        ```bash
        lspci | grep 4052
        # If there is only one card, the output may look like this; 82 is the card number. Multiple cards will show multiple entries
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
        - Configure the PCIE link. The following command must be run for each card:
        ```bash
        sudo setpci -v -s 82:*.0 ecap_acs+6.w=0
        ```
