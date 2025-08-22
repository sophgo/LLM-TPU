# prompt共享

## 模型编译

``` shell
# -s 指定总长度; --max_input_length 指定每次输入的最大长度; --max_prefill_kv_length 指定prompt共享的最大长度，会生成共用的kv cache传递给prefill
llm_convert.py -m /workspace/Qwen2.5-7B-Instruct-AWQ -s 8192 --quantize w4bf16 -c bm1684x --share_promt --max_input_length 512 --max_prefill_kv_length 4096 --out_dir qwen2.5_7b_share
```


## 运行
```shell
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
python3 pipeline.py -m ./qwen2.5-7bxxxx.bmodel -c ../config --prompt test.txt
```
