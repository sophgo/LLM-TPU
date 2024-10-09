# 序列共享demo

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
* **必须将total_seq比较大的模型放到model_path_list的前面**,也就是seq最大的那个先跑
* model_path_list：当使用多个模型时，用逗号隔开
* 权重复用的流程为：self.model = chat.Qwen() --> self.load_model(model_0) --> self.free_device --> self.load_model(model_1) --> self.model.deinit()

## 运行c-eval数据集
```shell
python3 pipeline.py --model_path encrypted.bmodel  --tokenizer_path ../support/token_config/ --devid 61 --generation_mode penalty_sample --lib_path ../../Qwen2/share_cache_demo/build/libcipher.so --max_new_tokens 50 --embedding_path embedding.bin
```

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

# 如何导出logits
如果您想查看每层输出的logits，您可以按照如下步骤来导出

## 1. clone cnpy库
```
mkdir third_party
cd third_party && git clone https://github.com/rogersce/cnpy.git
```

## 2. CMake编译时添加-DCMAKE_TYPE=DUMP
```shell
cd build && cmake -DCMAKE_TYPE=DUMP .. && make && cp *cpython* .. && cd ..
```

### 3. 修改chat.cpp文件
根据你需要查看的logits来写正确的代码，可以参考以下代码
```cpp
dump_tensor_to_file<uint16_t>(bm_handle,net_blocks[idx]->stages[0].output_mems[0],{1,6016,4096},"output_" + std::to_string(idx) + ".npz","hidden_states");
dump_tensor_to_file<uint16_t>(bm_handle,net_blocks[idx]->stages[0].output_mems[1],{1,6016,32,128},"output_" + std::to_string(idx) + ".npz","present_key");
dump_tensor_to_file<uint16_t>(bm_handle,net_blocks[idx]->stages[0].output_mems[2],{1,6016,32,128},"output_" + std::to_string(idx) + ".npz","present_value");
```
注意
* shape一定要设置正确，可以通过model_tool --info xxx.bmodel来查看shape
* 如果compile.sh转的是bf16类型，那么dump_tensor_to_file需要使用bf16_to_fp32_value；compile.sh转的是fp16类型，那么dump_tensor_to_file需要使用fp16_ieee_to_fp32_value

### 4. 导出npz文件
运行以下命令
```shell
rm *.npz *.onnx -f
python3 pipeline.py --model_path_list qwen-7b_int4_shareseq6016_1dev_dyn.bmodel --tokenizer_path ../support/token_config/ --devid 0 --generation_mode penalty_sample --mode debug
```

* 如果之前目录下有output_x.npz文件，记得提前删掉，不然会有问题
* 开启--mode debug模式来导出

### 5. 如何使用
```python
import numpy as np
x = np.load("output_0.npz")
print(x.files)
print(x["hidden_states"])
```


