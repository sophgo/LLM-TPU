# Environment Setup
```
pip3 install dfss
```

If you do not plan to compile the model yourself, you can directly use the downloaded model
```
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama2-7b_int4_1dev.bmodel
```

Compile the library files
```
mkdir build
cd build && cmake .. && make && cp llama2 .. && cd ..
```

# cpp demo
```
./llama2 --model llama2-7b_int4_1dev.bmodel --tokenizer ../support/token_config/tokenizer.model  --devid 0
```
