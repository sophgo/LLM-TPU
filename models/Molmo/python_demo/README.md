# 环境准备
> （python demo运行之前需要执行这个）
```
sudo apt-get update
sudo apt-get install pybind11-dev
```

如果不打算自己编译模型，可以直接用下载好的模型
```
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/molmo-7b_int4_seq1024_384x384.bmodel

编译库文件
```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

# python demo
```
python3 pipeline.py -m molmo-7b_int4_seq1024_384x384.bmodel -i ./test.jpg -s image_size -t ../processor_config --devid 0
```
