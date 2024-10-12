## python demo

### Install dependent
```bash
sudo apt-get update
pip3 install transformers_stream_generator einops tiktoken accelerate gradio transformers==4.45.2 
pip3 install pybind11[global]
```

### Compile chat.cpp

可以直接下载编译好的模型，不用自己编译
```bash
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-7b_int4_seq8192_1dev.bmodel
```

```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

### CLI demo
```bash
python3 pipeline.py --model_path your_bmodel_path --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy
```

### Gradio web demo
```bash
python3 web_demo.py --model_path your_bmodel_path
```
