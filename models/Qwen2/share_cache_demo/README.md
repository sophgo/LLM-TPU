# 序列共享demo

## 1. 编译模型
your_torch_model是你模型的路径
```shell
pip3 install torch==2.0.1 transformers_stream_generator einops tiktoken accelerate transformers==4.41.2
cp files/Qwen2-7B-Instruct/* /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/

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
python pipeline.py --model_path_list qwen2-7b_int4_share6144_unshare1536_1dev_encrypted.bmodel,qwen2-7b_int4_share6144_unshare1024_1dev_encrypted.bmodel,qwen2-7b_int4_share1248_unshare0_1dev_encrypted.bmodel  --tokenizer_path ../support/token_config/ --devid 0 --generation_mode penalty_sample --lib_path build/libcipher.so
```
* io_alone_mode：当io_alone_mode=0，则正常prefill；当io_alone_mode=1，则使用kvcache复用方案
* model_path_list：模型路径，当使用多个模型时，用逗号隔开
* lib_path：解密库路径，当使用加密模型时，必须带上lib_path参数，因为只有带上lib_path，才会走解密的逻辑

## 4. 运行c-eval测试
在pipeline.py中将test_ceval函数的注释删掉，之后运行
```
python3 pipeline.py --model_path encrypted.bmodel  --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy --lib_path build/libcipher.so --embedding_path embedding.bin --max_new_tokens 50
```

## 5. 注意事项

### 权重复用
* 如果使用权重复用的方案，在compile.sh完成后，可以使用以下指令来检查weight空间是否一致

```shell
model_tool --info qwen1.5-4b_int4_share6144_unshare2560_seq8704_1dev_dyn.bmodel | grep "weight"
model_tool --info qwen1.5-4b_int4_share6144_unshare2816_seq8960_1dev_dyn.bmodel | grep "weight"
```
> device mem size: 1680323988 (weight: 1050832896, instruct: 6612372, runtime: 622878720)
>
> device mem size: 1679614228 (weight: 1050832896, instruct: 5902612, runtime: 622878720)
>
> 他们的weight是一致的，都是1050832896，一点偏差也不能有，如果不一致，可能是下面这步没做
```shell
cp files/Qwen-7B-Chat/* your_torch_model
```

* 如果你想要权重复用，那么必须要用`free_device()`函数来释放空间，而不要用deinit()

### 模型加解密
* 建议使用定长加密，变长加密还没有经过测试
* 如果跑demo送入的是加密后的模型，必须要带上--lib_path参数，因为只有带上lib_path，才会走解密的逻辑
```shell
model_tool --encrypt -model origin.bmodel -net_name block_0 -lib ./build/libcipher.so -o encrypted.bmodel
```
