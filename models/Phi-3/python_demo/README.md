# Environment Setup
> (This must be executed before running the python demo)
```
sudo apt-get update
sudo apt-get install pybind11-dev
pip3 install transformers_stream_generator einops tiktoken accelerate transformers==4.39.3
pip3 install gradio==3.39.0 mdtex2html==1.2.0 dfss
```

If you do not plan to compile the phi3 model yourself, you can directly use the downloaded phi3 model
```
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/phi3-4b_int4_1dev.bmodel (original version)
```
or
```
python3 -m dfss --url=open@sophgo.com:/share/hengyang/phi-3-mini-4k-instruct_w4f16_seq512_bm1684x_1dev_20250811_151631.bmodel (version compiled with the new method)
```

Compile the library files
```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

# python demo
```
python3 pipeline.py --model_path compile/phi-xxx.bmodel --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy
```
