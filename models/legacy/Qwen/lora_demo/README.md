# lora_demo

## 1. Compile the model
your_torch_model is the path of your model

Note when compiling: **max_pos_len needs to be set to the largest total_seq among all models**
```shell
pip3 install transformers_stream_generator einops tiktoken accelerate transformers==4.32.0

cp files/Qwen-7B-Chat/* your_torch_model

./t.sh
```
If you do not want to compile the model, you can also download it directly
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen-7b_int4_shareseq6016_unshare1536_seq7552_1dev_dyn.bmodel
```

## 2. Compile the library files
```shell
sudo apt-get install libcrypto++-dev libcrypto++-doc libcrypto++-utils

mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

## 3. Run the python demo
```shell
python3 pipeline.py --model_path encrypted.bmodel  --tokenizer_path ../support/token_config/ --devid 23 --generation_mode penalty_sample --lib_path ../../Qwen2/share_cache_demo/build/libcipher.so --embedding_path embedding.bin
```
* For other usage details, please see models/Qwen/share_cache_demo/README.md