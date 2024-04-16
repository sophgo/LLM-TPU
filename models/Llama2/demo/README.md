# Llama2-13B
目前支持6芯int4/int8
## 1. 安装驱动
按如下命令下载并安装驱动：
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/libsophon-0415deb.tar.gz
tar -xzf libsophon-0415deb.tar.gz
cd libsophon-0415deb
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

按如下操作编译模型：
```shell
cd compile
python3 ./export_onnx_13b.py -m path_to/Llama2-13B-Chat/
./compile_13B.sh --mode int4 --name llama2-13b --num_device 6 --seq_length 512
```

如果不打算编译模型，可以通过以下命令下载已编译好的模型：
```shell
pip3 install dfss
# int4 bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama2-13B/llama2-13b_int4_6dev.bmodel
# int8 bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama2-13B/llama2-13b_int8_6dev.bmodel
```

## 3. 运行
```shell
cd demo
mkdir build && cd build
cmake .. && make -j8
./llama2_parallel --model path_to/llama2-13b_int4_6dev.bmodel --devid 0,1,2,3,4,5 --tokenizer path_to/tokenizer.model
```
