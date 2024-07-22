# Command

### export onnx + combine bmodel
请见Qwen1_5/compile/README.md

### 下载迁移好的模型
也可以直接下载编译好的模型，不用自己编译
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-1.8b_int4_1dev_seq512.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-1.8b_int8_1dev_seq512.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-1.8b_int4_1dev_seq1280.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-1.8b_int8_1dev_seq1280.bmodel
```

### python demo

```shell
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..

python3 pipeline.py --model_path qwen1.5-1.8b_int4_1dev_seq512.bmodel --tokenizer_path ../token_config/ --devid 0 --generation_mode penalty_sample
```
