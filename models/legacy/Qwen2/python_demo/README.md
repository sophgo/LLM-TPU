## python demo

### Install dependent
```bash
sudo apt-get update
pip3 install transformers_stream_generator einops tiktoken accelerate gradio transformers==4.41.2 
pip3 install pybind11[global]
```

### encrypt bmodel

```bash
model_tool --encrypt -model qwen2-1.5b_bf16_seq4096_1dev.bmodel -net_name block_0 -lib libcipher.so -o qwen2-1.5b_bf16_seq4096_1dev_encrypted.bmodel
```
* -model inputs the combined model or a normal bmodel, -net inputs the network to be encrypted, -lib implements the specific encryption algorithm, and -o outputs the name of the encrypted model
* It can encrypt the model weights, flatbuffer structured data, and header.
* The encryption/decryption interfaces must be implemented in C style; C++ is not allowed. The interfaces are defined as follows:
```c
extern "C" uint8_t* encrypt(const uint8_t* input, uint64_t input_bytes, uint64_t* output_bytes);
extern "C" uint8_t* decrypt(const uint8_t* input, uint64_t input_bytes, uint64_t* output_bytes);
```

### Compile chat.cpp

You can directly download the pre-compiled model instead of compiling it yourself
```bash
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2-7b_int4_seq8192_1dev.bmodel
```

```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

### CLI demo
```bash
python3 pipeline.py --model_path your_bmodel_path --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy
```

### run encrypted bmodel demo
```bash
python3 pipeline.py --model_path your_bmodel_path --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy --lib_path libcipher.so
```

### test c-eval
```bash
python3 pipeline_checked.py --model_path ../compile/qwen2-1.5b_bf16_seq4096_1dev_encrypted.bmodel --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy --lib_path ../share_cache_demo/build/libcipher.so --max_new_tokens 50
```

### Gradio web demo
```bash
python3 web_demo.py --model_path your_bmodel_path
```
