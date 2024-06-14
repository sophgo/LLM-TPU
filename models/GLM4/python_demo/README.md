# 环境准备
> （python demo和web demo运行之前都需要执行这个）
```
sudo apt-get update
sudo apt-get install pybind11-dev
pip3 install sentencepiece transformers==4.30.2
pip3 install gradio==3.39.0 mdtex2html==1.2.0 dfss
```

如果不打算自己编译模型，可以直接用下载好的模型
```
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/chatglm3-6b_int4_1dev_512.bmodel
```

编译库文件
```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

# python demo
```
python3 pipeline.py --model_path glm4-9b_int4_1dev.bmodel --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy
```

# web demo
```
python3 web_demo.py --model_path glm4-9b_int4_1dev.bmodel --tokenizer_path ../support/token_config/ --devid 0
```
