### python demo

For the python demo, be sure to source envsetup.sh inside LLM-TPU (it is different from the envsetup.sh in tpu-mlir).
```shell
cd /workspace/LLM-TPU
source envsetup.sh
```

```shell
pip3 install pybind11[global] transformers_stream_generator einops tiktoken accelerate transformers==4.32.0
```

Download the migrated models:
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/codefuse-7b_int4_1dev_2048.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/codefuse-7b_int8_1dev_2048.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/codefuse-7b_int4_1dev_4096.bmodel
```

Run the demo:
```shell
cd /workspace/LLM-TPU/models/CodeFuse/python_demo
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..

python3 pipeline.py --model_path codefuse-7b_int4_1dev.bmodel --tokenizer_path ../token_config/ --devid 0 --generation_mode greedy
```
