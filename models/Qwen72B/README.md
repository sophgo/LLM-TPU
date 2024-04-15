# Qwen72B
相关操作说明详细可见[Qwen](../Qwen)。

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
驱动安装好后，便可以在直接执行`./run_demo.sh`来运行本demo。


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
cd Qwen72B/compile
python3 ./export_onnx.py -m path_to/Qwen-72B-Chat/ --num_threads 72
./compile.sh --mode int4 --num_device 8 --addr_mode io_alone --seq_length 8192
```

如果不打算编译模型，可以通过以下命令下载已编译好的模型：
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/qwen-72b/qwen-72b_int4_8192_8dev.bmodel
```

## 3. 运行
```shell
cd demo
mkdir build && cd build
cmake .. && make -j8
./qwen --model path_to/qwen-72b_int4_8192_8dev.bmodel --devid 0,1,2,3,4,5,6,7 --tokenizer path_to/qwen.tiktoken
```
