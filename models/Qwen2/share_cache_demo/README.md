# Sequence Sharing Demo

## 1. Compile the Model
your_torch_model is the path to your model
```shell
pip3 install torch==2.0.1 transformers_stream_generator einops tiktoken accelerate transformers==4.41.2
cp files/Qwen2-7B-Instruct/* /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/

./t.sh
```
If you don't want to compile the model, you can also download it directly
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen-7b_int4_shareseq6016_unshare1536_seq7552_1dev_dyn.bmodel
```


## 2. Compile the Library Files
```shell
sudo apt-get install libcrypto++-dev libcrypto++-doc libcrypto++-utils

mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

## 3. Run the Python Demo
```shell
python3 pipeline.py --model_path encrypted.bmodel  --tokenizer_path ../support/token_config/ --devid 0 --generation_mode penalty_sample --lib_path build/libcipher.so --embedding_path embedding.bin
```
* io_alone_mode: when io_alone_mode=0, normal prefill is performed; when io_alone_mode=1, the KV Cache reuse scheme is used
* model_path_list: model paths; when using multiple models, separate them with commas
* lib_path: path to the decryption library. When using an encrypted model, you must pass the lib_path parameter, because only with lib_path will the decryption logic be invoked

## 4. Run the C-Eval Test
In pipeline.py, remove the comment on the engine.test_ceval() function, then run
```
mkdir ceval-exam 
cd ceval-exam
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/ceval-exam.zip
unzip ceval-exam

cp /path_to/LLM-TPU/harness/C-Eval/subject_mapping.json

python3 pipeline.py --model_path encrypted.bmodel  --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy --lib_path build/libcipher.so --embedding_path embedding.bin --max_new_tokens 50 | tee test_ceval.log
```

## 5. Notes

### Weight Reuse
* If you use the weight reuse scheme, after compile.sh completes, you can use the following commands to check whether the weight space is consistent

```shell
model_tool --info qwen1.5-4b_int4_share6144_unshare2560_seq8704_1dev_dyn.bmodel | grep "weight"
model_tool --info qwen1.5-4b_int4_share6144_unshare2816_seq8960_1dev_dyn.bmodel | grep "weight"
```
> device mem size: 1680323988 (weight: 1050832896, instruct: 6612372, runtime: 622878720)
>
> device mem size: 1679614228 (weight: 1050832896, instruct: 5902612, runtime: 622878720)
>
> Their weights are identical, both 1050832896; there must be no deviation at all. If they are inconsistent, the following step may have been skipped
```shell
cp files/Qwen-7B-Chat/* your_torch_model
```

* If you want weight reuse, you must use the `free_device()` function to release the space instead of deinit()

### Model Encryption/Decryption
* Fixed-length encryption is recommended; variable-length encryption has not been tested yet
* If the model fed to the demo is encrypted, you must pass the --lib_path parameter, because only with lib_path will the decryption logic be invoked
```shell
model_tool --encrypt -model origin.bmodel -net_name block_0 -lib ./build/libcipher.so -o encrypted.bmodel
```

### Reducing Log Output
* If you want to reduce log output such as `Can't find network name`, you can run the following command
```shell
export BMRT_LOG_VERSION=3
```
