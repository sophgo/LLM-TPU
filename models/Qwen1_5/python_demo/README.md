# Command

### export onnx + combine bmodel
Please see Qwen1_5/compile/README.md

### Download the migrated model
You can also directly download the pre-compiled model instead of compiling it yourself
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
