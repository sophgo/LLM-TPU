# Export onnx
```
cp ../compile/files/Qwen-7B-Chat/* your_torch_path
python export_onnx.py --model_path your_torch_path --device cpu --num_threads 8 --max_prompt_length 6144 --seq_length 8192
```
Here max_prompt_length refers to the maximum length of the prompt, and seq_length refers to the maximum length of the input plus the output

# Compile bmodel
```
./compile.sh --mode int4 --name qwen-7b --seq_length 8192 --addr_mode io_alone
```

# Environment Setup
> (This must be executed before running either the python demo or the web demo)
```
sudo apt-get update
sudo apt-get install pybind11-dev
pip3 install transformers_stream_generator einops tiktoken accelerate transformers==4.32.0
```

If you do not plan to compile the model yourself, you can directly use the downloaded model
```
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen-7b_int4_seq8192_1dev_prompt_cache.bmodel
```

Compile the library files
```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

# python demo
```
python3 pipeline.py --model_path qwen-7b_int4_seq8192_1dev_prompt_cache.bmodel --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy
```