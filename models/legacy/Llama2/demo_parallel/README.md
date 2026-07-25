# Llama2-13B
Currently supports 4/6/8-chip int4/int8
## 1. Install the Driver
Download and install the driver with the following commands. **Note that you must use this version of the driver; older driver versions do not support the latest multi-chip models**:
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/drivers/libsophon-0425deb.tar.gz
tar -xzf libsophon-0425deb.tar.gz
cd libsophon-0425deb
sudo apt remove sophon-driver sophon-libsophon
sudo dpkg -i *.deb

```
After the driver is installed, you can directly execute `./run_demo_parallel.sh` to run this demo.

## 2. Compile
Model compilation needs to be performed in docker. Enter docker with the following commands:
```shell
docker pull sophgo/tpuc_dev:latest
# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```

Compile the model as follows. Currently the multi-chip demo only supports the lmhead_with_topk case; set it to 1 when compiling to export only the model for this case:
```shell
cd compile
python3 ./export_onnx.py --model_path path_to/Llama2-13B-Chat/ --lmhead_with_topk 1 --seq_length 512
./compile.sh --mode int4 --name llama2-13b --num_device 6 --seq_length 512
```

If you do not plan to compile the model, you can download the precompiled models with the following commands. The following models are currently precompiled. **Note that the latest driver version requires re-downloading the models below**:
```shell
pip3 install dfss
# int4 bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/llama2-13b_int4_8dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/0529/llama2-13b_int4_6dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/llama2-13b_int4_4dev.bmodel
# int8 bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/llama2-13b_int8_8dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/0529/llama2-13b_int8_6dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/llama2-13b_int8_4dev.bmodel
```

## 3. Run
```shell
cd demo
mkdir build && cd build
cmake .. && make -j8
./llama2_parallel --model path_to/llama2-13b_int4_6dev.bmodel --devid 0,1,2,3,4,5 --tokenizer path_to/tokenizer.model
```

## 4. FAQ

1) If the model loading process is interrupted suddenly for no apparent reason, you can run `ulimit -HSn 65536` to increase system resources

2) If the model performance is much worse than expected or it hangs during runtime, run `test_cdma_p2p 0x130000000 0 0x140000000 1 0x100000` to test p2p performance. If the bandwidth is only around 1500MB/s, p2p may be unavailable. Enable it with the following steps:
    - iommu is not disabled; disable it with the following procedure:
      ```bash
      sudo vi /etc/default/grub
      # Add intel_iommu=off/amd_iommu=off according to the CPU type
      # GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet intel_iommu=off"
      # GRUB_CMDLINE_LINUX="crashkernel=auto rhgb quiet amd_iommu=off"
      sudo update-grub
      sudo reboot
      ```
    - If the speed still does not improve after disabling iommu, you may also need to configure the PCIE link. Run the following commands, then reinstall the driver:
      ```bash
      sudo setpci -v -s 99:*.0 ecap_acs+6.w=0
      sudo setpci -v -s 32:*.0 ecap_acs+6.w=0
      ```
