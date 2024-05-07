# Command

## Export onnx

```shell
pip install sentencepiece transformers==4.30.2
```

### export onnx
../compile/chatglm3-6b是你torch模型的位置
```shell
cp files/chatglm3-6b/modeling_chatglm.py ../compile/chatglm3-6b

python export_onnx.py --model_path ../compile/chatglm3-6b --device cpu --seq_length 512 --num_threads 8
```

## Compile bmodel

```shell
pushd /path_to/tpu-mlir
source envsetup.sh
popd
```

### compile bmodel
```shell
./compile.sh --mode int8 --name chatglm3-6b --num_device 2
```

也可以直接下载编译好的模型，不用自己编译
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/chatglm3-6b_int8_2dev_512.bmodel
```

## python demo
首先准备环境
```shell
sudo pip3 install pybind11[global] sentencepiece
```
之后编译库文件，并运行
```shell
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..

python3 pipeline.py --model_path chatglm3-6b_int8_2dev_512.bmodel --tokenizer_path ../support/token_config/ --devid 0,1 --generation_mode greedy
```
