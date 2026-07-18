# lora demo

## 1. Compile the model
your_torch_model is the path of your model
```shell
./run_compile.sh
```

## 2. Compile the library files
```shell
sudo apt-get install libcrypto++-dev libcrypto++-doc libcrypto++-utils

mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

## 3. Run the python demo
```shell
python3 pipeline.py --model_path encrypted.bmodel  --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy --lib_path ../share_cache_demo/build/libcipher.so --embedding_path embedding.bin --lora_path encrypted_lora_weights.bin --enable_lora_embedding
```
* lora_path: path to the decryption library. When using an encrypted model, the lib_path parameter must be provided, because the decryption logic is only executed when lib_path is provided.

## 4. Test command
```shell
./test_pipeline.py | tee test_pipeline.log
```
On the SoC, you also need to test the following command. For details, see Qwen2/lora_demo/test_pipeline.sh
```
python3 pipeline.py \
    --model_path encrypted.bmodel \
    --tokenizer_path ../support/token_config/ \
    --devid 0 \
    --generation_mode greedy \
    --lib_path ../share_cache_demo/build/libcipher.so \
    --embedding_path embedding.bin \
    --lora_path encrypted_lora_weights.bin \
    --zero_lora_path encrypted_lora_weights_0_0.bin \
    --enable_lora_embedding
```
* test_block.py can only be run on a server with a BM1684X board


### Reduce log output
* If you want to reduce log output such as `Can't find network name`, you can execute the following command
```shell
export BMRT_LOG_VERSION=3
```
