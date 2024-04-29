# onnx导出
```
cp ../compile/files/Qwen-7B-Chat/* your_torch_path
python export_onnx.py --model_path your_torch_path --device cpu --num_threads 8 --max_prompt_length 4096 --seq_length 8192
```
这里的max_prompt_length指的是prompt的最大长度，seq_length指的是输入加输出之和的最大长度

# bmodel编译
```
./compile.sh --mode int4 --name qwen-7b --seq_length 8192 --addr_mode io_alone
```

# 环境准备
> （python demo和web demo运行之前都需要执行这个）
```
sudo apt-get update
sudo apt-get install pybind11-dev
pip3 install transformers_stream_generator einops tiktoken accelerate transformers==4.32.0
```

如果不打算自己编译模型，可以直接用下载好的模型
```
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen-7b_int4_seq8192_1dev_prompt_cache.bmodel
```

编译库文件
```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

# python demo
```
python3 pipeline.py --model_path qwen-7b_int4_seq8192_1dev_prompt_cache.bmodel --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy
```