# Sequence sharing demo

## 1. Compile the model
your_torch_model is the path to your model

Note when compiling: **max_pos_len needs to be set to the largest total_seq among all models**
```shell
pip3 install transformers_stream_generator einops tiktoken accelerate transformers==4.32.0

cp files/Qwen-7B-Chat/* your_torch_model

./t.sh
```
If you don't want to compile the model, you can also download it directly
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen-7b_int4_shareseq6016_unshare1536_seq7552_1dev_dyn.bmodel
```

## 2. Build the library files
```shell
sudo apt-get install libcrypto++-dev libcrypto++-doc libcrypto++-utils

mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

## 3. Run the python demo
```shell
python3 pipeline.py --model_path encrypted.bmodel  --tokenizer_path ../support/token_config/ --devid 23 --generation_mode penalty_sample --lib_path ../../Qwen2/share_cache_demo/build/libcipher.so --embedding_path embedding.bin
```
* **The model with the larger total_seq must be placed first in model_path_list**, i.e. the one with the largest seq runs first
* model_path_list: when using multiple models, separate them with commas
* The weight reuse flow is: self.model = chat.Qwen() --> self.load_model(model_0) --> self.free_device --> self.load_model(model_1) --> self.model.deinit()

## Run the c-eval dataset
```shell
python3 pipeline.py --model_path encrypted.bmodel  --tokenizer_path ../support/token_config/ --devid 61 --generation_mode penalty_sample --lib_path ../../Qwen2/share_cache_demo/build/libcipher.so --max_new_tokens 50 --embedding_path embedding.bin
```

## 4. Notes

### Memory settings

Use the following to set the memory. The current memory usage is 10483MB, and the configured memory is 10512MB.
```shell
./memory_edit.sh -c -npu 6462 -vpu 0 -vpp 4050
```

### Weight reuse
* If you use the weight reuse scheme, after compile.sh completes, you can use the following commands to check whether the weight space is consistent

```shell
model_tool --info qwen1.5-4b_int4_share6144_unshare2560_seq8704_1dev_dyn.bmodel | grep "weight"
model_tool --info qwen1.5-4b_int4_share6144_unshare2816_seq8960_1dev_dyn.bmodel | grep "weight"
```
> device mem size: 1680323988 (weight: 1050832896, instruct: 6612372, runtime: 622878720)
>
> device mem size: 1679614228 (weight: 1050832896, instruct: 5902612, runtime: 622878720)
>
> Their weights are identical, both 1050832896, without the slightest deviation. If they are inconsistent, the following step was probably missed
```shell
cp files/Qwen-7B-Chat/* your_torch_model
```

### Model encryption/decryption
* Remember to run sudo apt-get install libcrypto++-dev libcrypto++-doc libcrypto++-utils
* If you use the model decryption scheme, **it is recommended to back up the original model in advance**, because the flatbuffer of the original model will be rewritten in place
* An example of model encryption/decryption is shown below; you only need to pass in the bmodel path. For details, refer to pipeline.py
```python
self.model.encrypt_bmodel(self.model_list[1])
```

# How to export logits
If you want to view the logits output by each layer, you can export them by following the steps below

## 1. Clone the cnpy library
```
mkdir third_party
cd third_party && git clone https://github.com/rogersce/cnpy.git
```

## 2. Add -DCMAKE_TYPE=DUMP when building with CMake
```shell
cd build && cmake -DCMAKE_TYPE=DUMP .. && make && cp *cpython* .. && cd ..
```

### 3. Modify the chat.cpp file
Write the correct code according to the logits you want to view; you can refer to the following code
```cpp
dump_tensor_to_file<uint16_t>(bm_handle,net_blocks[idx]->stages[0].output_mems[0],{1,6016,4096},"output_" + std::to_string(idx) + ".npz","hidden_states");
dump_tensor_to_file<uint16_t>(bm_handle,net_blocks[idx]->stages[0].output_mems[1],{1,6016,32,128},"output_" + std::to_string(idx) + ".npz","present_key");
dump_tensor_to_file<uint16_t>(bm_handle,net_blocks[idx]->stages[0].output_mems[2],{1,6016,32,128},"output_" + std::to_string(idx) + ".npz","present_value");
```
Note
* The shape must be set correctly; you can check the shape with model_tool --info xxx.bmodel
* If compile.sh converted to the bf16 type, then dump_tensor_to_file needs to use bf16_to_fp32_value; if compile.sh converted to the fp16 type, then dump_tensor_to_file needs to use fp16_ieee_to_fp32_value

### 4. Export the npz files
Run the following command
```shell
rm *.npz *.onnx -f
python3 pipeline.py --model_path_list qwen-7b_int4_shareseq6016_1dev_dyn.bmodel --tokenizer_path ../support/token_config/ --devid 0 --generation_mode penalty_sample --mode debug
```

* If there are output_x.npz files in the directory from before, remember to delete them in advance, otherwise problems will occur
* Enable --mode debug to export

### 5. How to use
```python
import numpy as np
x = np.load("output_0.npz")
print(x.files)
print(x["hidden_states"])
```

