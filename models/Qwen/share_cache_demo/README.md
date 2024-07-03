# 序列共享demo

## 1. 编译模型
your_torch_model是你模型的路径
```shell
pip3 install transformers_stream_generator einops tiktoken accelerate transformers==4.32.0

cp files/Qwen-7B-Chat/* your_torch_model

python export_onnx.py --model_path your_torch_model --device cpu --share_length 5888 --unshare_length 1536 --seq_length 8704 --num_thread 16

python export_onnx.py --model_path your_torch_model --device cpu --share_length 6016 --unshare_length 1024 --seq_length 7552 --max_pos_len 8704 --num_thread 16

./compile.sh --mode int4 --name qwen-7b --share_length 5888 --addr_mode io_alone --unshare_length 1536 --dynamic 1

./compile.sh --mode int4 --name qwen-7b --share_length 6016 --addr_mode io_alone --unshare_length 1024 --dynamic 1
```
如果你不想编译模型，也可以直接下载
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen-7b_int4_shareseq6016_unshare1536_seq7552_1dev_dyn.bmodel
```
* 使用的TPU-MLIR版本： bacc66292743153ff2f16927bffee69ffacb476c
* 内存：9663MB（动态）

# 分片方式
|第一片                  |第二片                 |第三片              |总长度              |
|:-                     |:-                     |:-                 |:-                 |
|share                  |unshare                |decode             |seq                |
|share_length=6016      |unshare_length=1536    |decode_length=0    |seq_length=7552    |

## 2. 编译库文件
```shell
sudo apt-get install libcrypto++-dev libcrypto++-doc libcrypto++-utils

mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

## 3. 运行python demo
```shell
python3 pipeline.py --model_path qwen-7b_int4_share5888_unshare1536_1dev_dyn.bmodel,qwen-7b_int4_share6016_unshare1024_1dev_dyn.bmodel  --tokenizer_path ../support/token_config/ --devid 0 --generation_mode penalty_sample --memory_prealloc --is_decrypt
```
* io_alone_reuse：使用io_alone_reuse时，表示上次的past_kv与io空间会复用，如果想要复用prefill，必须要io_alone_reuse=True
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
rm *.npz
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


