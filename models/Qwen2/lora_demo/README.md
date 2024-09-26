# lora demo

## 1. 编译模型
your_torch_model是你模型的路径
```shell
./run_compile.sh
```

## 2. 编译库文件
```shell
sudo apt-get install libcrypto++-dev libcrypto++-doc libcrypto++-utils

mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

## 3. 运行python demo
```shell
python3 pipeline.py --model_path encrypted.bmodel  --tokenizer_path ../support/token_config/ --devid 0 --generation_mode penalty_sample --lib_path ../share_cache_demo/build/libcipher.so --embedding_path embedding.bin --lora_path lora_weights.bin
```
* lora_path：解密库路径，当使用加密模型时，必须带上lib_path参数，因为只有带上lib_path，才会走解密的逻辑


### 减少日志打印
* 如果想要减少类似`Can't find network name`这种日志打印，可以执行如下命令
```shell
export BMRT_LOG_VERSION=3
```
