# 环境准备
```
pip3 install dfss
```

如果不打算自己编译模型，可以直接用下载好的模型
```
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama2-7b_int4_1dev.bmodel
```

编译库文件
```
mkdir build
cd build && cmake .. && make && cp llama2 .. && cd ..
```

# cpp demo
```
./llama2 --model llama2-7b_int4_1dev.bmodel --tokenizer ../support/token_config/tokenizer.model  --devid 0
```
