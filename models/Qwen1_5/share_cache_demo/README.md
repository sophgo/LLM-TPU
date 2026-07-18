# Sequence sharing demo

## 1. Compile the model
`your_torch_model` is your torch model; `--dynamic 1` means prefill uses dynamic compilation.
```shell
pip3 install transformers==4.37.0

cp files/Qwen1.5-4B-Chat/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/

python export_onnx.py --model_path your_torch_model --device cpu --share_length 6144 --unshare_length 2560 --seq_length 8704 --num_thread 16

./compile.sh --mode int4 --name qwen1.5-4b --share_length 6144 --addr_mode io_alone --unshare_length 2560 --dynamic 1
```
If you do not want to compile the model, you can also download it directly
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-4b_int4_shareseq6144_unshare2560_seq8704_1dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-4b_int4_shareseq6144_unshare2560_seq8704_1dev_dyn.bmodel
```
* TPU-MLIR version used: bacc66292743153ff2f16927bffee69ffacb476c
* Runtime memory: 6958MB (dynamic)

## Segmentation scheme
|First segment          |Second segment         |Third segment      |Total length       |
|:-                     |:-                     |:-                 |:-                 |
|share                  |unshare                |decode             |seq                |
|share_length=6144      |unshare_length=2560    |decode_length=0    |seq_length=8704    |


## 2. Compile the library
```shell
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```


## 3. Run the python demo
```shell
python3 pipeline.py --model_path_list qwen1.5-4b_int4_shareseq6144_unshareseq2816_seq8960_1dev_dyn.bmodel,qwen1.5-4b_int4_shareseq6144_unshareseq2560_seq8704_1dev_dyn.bmodel --tokenizer_path ../token_config/ --devid 0 --generation_mode penalty_sample --memory_prealloc --is_decrypt
```
* memory_prealloc: enables weight reuse
* is_decrypt: enables model decryption. **Currently, memory_prealloc and is_decrypt can only be used together**
* model_path_list: when using multiple models, separate them with commas
* The weight reuse flow is: self.model = chat.Qwen() --> self.load_model(model_0) --> self.free_device --> self.load_model(model_1) --> self.model.deinit()
* If the two models have different weights, for example one Qwen-7B and one Qwen1.5-4B, it is recommended to recreate the class, i.e. self.model = chat.Qwen --> self.model.deinit() --> self.model = chat.Qwen --> self.model.deinit()


## 4. Notes

When using weight reuse or shared-sequence reuse, it is recommended to load the bmodel with the maximum length first, otherwise the following error is likely to occur
```
[bmlib_memory][error] bm_alloc_gmem failed, dev_id = 0, size = 0x3060e078
[BM_CHECK][error] BM_CHECK_RET fail /workspace/libsophon/bmlib/src/bmlib_memory.cpp: bm_malloc_device_byte_u64: 1054
[BMRT][alloc_device_mem_u64:3028] FATAL:Error: device memory: neuron_mem don't alloc
```
That is, **recommended**
```python
python3 pipeline.py --model_path_list qwen-7b_int4_share6016_unshare1536_seq8704_1dev_dyn.bmodel,qwen-7b_int4_share5888_unshare1024_seq8704_1dev_dyn.bmodel  ...
```
**Strongly not recommended**
```python
python3 pipeline.py --model_path_list qwen-7b_int4_share5888_unshare1024_seq8704_1dev_dyn.bmodel,qwen-7b_int4_share6016_unshare1536_seq8704_1dev_dyn.bmodel  ...
```
The purpose of this is to allocate the largest runtime space (neuron space) first

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
> Their weights are identical, both 1050832896; not even the slightest deviation is allowed. If they are inconsistent, the following step may have been skipped
```shell
cp files/Qwen1.5-4B-Chat/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/
```

### Model encryption/decryption
* Remember to run sudo apt-get install libcrypto++-dev libcrypto++-doc libcrypto++-utils
* If you use the model decryption scheme, **it is recommended to back up the original model in advance**, because the flatbuffer of the original model will be rewritten in place
* An example of model encryption/decryption is shown below; you only need to pass in the bmodel path. See pipeline.py for details
```python
self.model.encrypt_bmodel(self.model_list[1])
```
