# Multi-Chip Demo

## 1. Install the Driver
Download and install the driver with the following commands. **Note: this specific driver version is currently required; older driver versions do not support the latest multi-chip models**:
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/drivers/libsophon-0523deb.tar.gz
tar -xzf libsophon-0523deb.tar.gz
cd libsophon-0523deb
sudo apt remove sophon-driver sophon-libsophon
sudo dpkg -i *.deb
```
After the driver is installed, you can directly run `./run_demo_parallel.sh` to run this demo.


## 2. Compile
Compiling the model must be done in docker. Enter docker with the following commands:
```shell
docker pull sophgo/tpuc_dev:latest
# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v /dev:/dev -v /opt/sophon:/opt/sophon -v /etc/profile.d:/etc/profile.d -v /etc/ld.so.conf.d:/etc/ld.so.conf.d -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```

Compile the model as follows:
```shell
cd Qwen/compile
python3 ./export_onnx.py -m path_to/Qwen-72B-Chat/ --num_threads 72 --lmhead_with_topk 1
# static compilation
./compile.sh --mode int4 --num_device 8 --addr_mode io_alone --seq_length 8192
# dynamic compilation
./compile.sh --mode int4 --num_device 8 --addr_mode io_alone --seq_length 8192 --dynamic 1
```

If you do not plan to compile the model, you can download the pre-compiled models with the following commands. The following models are currently pre-compiled. **Note: the latest driver version requires re-downloading the models below**:
```shell
pip3 install dfss
# int4
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen-72b_int4_8192_8dev_dyn.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen-14b_int4_8dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen-72b_int4_8192_8dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen-14b_int4_8dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/0529/qwen-14b_int4_6dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen-14b_int4_4dev.bmodel
# int8
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen-14b_int8_8dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/0529/qwen-14b_int8_6dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/qwen-14b_int8_4dev.bmodel
```

## 3. Run
```shell
git submodule update --init

cd demo
mkdir build && cd build
cmake .. && make -j8
./qwen --model path_to/qwen-72b_int4_8192_8dev.bmodel --devid 0,1,2,3,4,5,6,7 --tokenizer path_to/qwen.tiktoken
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
        - Run the following command to determine the card number:
        ```bash
        lspci | grep 4052
        # If there is only one card, the output may look like the following; 82 is the card number. Multiple cards will show multiple entries.
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
        - Configure the PCIE link; run the following command for each card:
        ```bash
        sudo setpci -v -s 82:*.0 ecap_acs+6.w=0
        ```
