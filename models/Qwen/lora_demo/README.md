# lora_demo

## 1. 编译模型
your_torch_model是你模型的路径

在编译的时候需要注意：**max_pos_len需要设置为所有模型中最大的total_seq**
```shell
pip3 install transformers_stream_generator einops tiktoken accelerate transformers==4.32.0

cp files/Qwen-7B-Chat/* your_torch_model

./t.sh
```
如果你不想编译模型，也可以直接下载
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen-7b_int4_shareseq6016_unshare1536_seq7552_1dev_dyn.bmodel
```

## 2. 编译库文件
```shell
sudo apt-get install libcrypto++-dev libcrypto++-doc libcrypto++-utils

mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

## 3. 运行python demo
```shell
python3 pipeline.py --model_path encrypted.bmodel  --tokenizer_path ../support/token_config/ --devid 23 --generation_mode penalty_sample --lib_path ../../Qwen2/share_cache_demo/build/libcipher.so --embedding_path embedding.bin
```
* 其他使用细节，请见models/Qwen/share_cache_demo/README.md