# 环境准备
> （python demo运行之前需要执行这个）
```
sudo apt-get update
sudo apt-get install pybind11-dev
pip install einops torchvision transformers==4.43.3
```

如果不打算自己编译模型，可以直接用下载好的模型
```
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/molmo-7b_int4_2048seq.bmodel

编译库文件
```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

# python demo
```
python3 pipeline.py --model_path molmo-7b_int4_2048seq.bmodel --image_path ./test.jpg --tokenizer_path ../token_config/ --devid 0 --generation_mode greedy
```
