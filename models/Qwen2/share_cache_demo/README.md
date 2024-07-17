# 序列共享demo

## 1. 编译模型
your_torch_model是你模型的路径
```shell
pip install transformers_stream_generator einops tiktoken accelerate transformers==4.41.2
cp files/Qwen2-7B-Instruct/* /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/

python export_onnx.py --model_path your_torch_model --device cpu --share_length 6016 --unshare_length 1536 --seq_length 8704 --num_thread 16 --max_pos_len 8704

python export_onnx.py --model_path your_torch_model --device cpu --share_length 6016 --unshare_length 1024 --seq_length 7552 --max_pos_len 8704 --num_thread 16

./compile.sh --mode int4 --name qwen2-7b --share_length 5888 --addr_mode io_alone --unshare_length 1536
./compile.sh --mode int4 --name qwen-7b --share_length 6016 --addr_mode io_alone --unshare_length 1024
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
python3 pipeline_test.py --model_path_list qwen2-7b_int4_share6144_unshare1024_1dev_test.bmodel,qwen2-7b_int4_share6144_unshare1024_1dev_test.bmodel  --tokenizer_path ../support/token_config/ --devid 18 --generation_mode greedy --memory_prealloc --is_decrypt
```
* memory_prealloc：表示使用权重复用
* is_decrypt：表明使用模型解密，**目前仅支持memory_prealloc和is_decrypt同时使用**
* model_path_list：当使用多个模型时，用逗号隔开
* 权重复用的流程为：self.model = chat.Qwen() --> self.load_model(model_0) --> self.free_device --> self.load_model(model_1) --> self.model.deinit()
* 如果两个模型权重不一致，比如一个Qwen-7B 一个Qwen1.5-4B，那么建议重新创建一个类，即 self.model = chat.Qwen --> self.model.deinit() --> self.model = chat.Qwen --> self.model.deinit()


## 4. 注意事项

### 内存设置

使用如下方式来设定内存，目前内存占用为10483MB，设定的内存为10512MB
```shell
./memory_edit.sh -c -npu 6462 -vpu 0 -vpp 4050
```

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

### 模型加解密
* 记得使用sudo apt-get install libcrypto++-dev libcrypto++-doc libcrypto++-utils
* 如果使用模型解密的方案，**建议提前备份好原始模型**，因为会直接原地改写原始模型的flatbuffer
* 模型加解密的实例如下所示，只需要传入bmodel路径即可，具体请参考pipeline.py
```python
self.model.encrypt_bmodel(self.model_list[1])
```
