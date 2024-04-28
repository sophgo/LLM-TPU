# Llama2-13B
目前支持4/6/8芯int4/int8
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

按如下操作编译模型，目前多芯demo仅支持lmhead_with_topk的情况，编译时设置为1，只导出该情况的模型：
```shell
cd compile
python3 ./export_onnx.py --model_path path_to/Llama2-13B-Chat/ --lmhead_with_topk 1 --seq_length 512
./compile.sh --mode int4 --name llama2-13b --num_device 6 --seq_length 512
```

如果不打算编译模型，可以通过以下命令下载已编译好的模型，目前有如下模型已经预编译好，**注意最新版本的驱动需要重新下载下方的模型**：
```shell
pip3 install dfss
# int4 bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/llama2-13b_int4_8dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/llama2-13b_int4_6dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/llama2-13b_int4_4dev.bmodel
# int8 bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/llama2-13b_int8_8dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/llama2-13b_int8_6dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/llama2-13b_int8_4dev.bmodel
```

## 3. 运行
```shell
cd demo
mkdir build && cd build
cmake .. && make -j8
./llama2_parallel --model path_to/llama2-13b_int4_6dev.bmodel --devid 0,1,2,3,4,5 --tokenizer path_to/tokenizer.model
```
