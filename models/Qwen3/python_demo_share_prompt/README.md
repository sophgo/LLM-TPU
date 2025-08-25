# prompt共享

可以将较长的prompt生成kv cache，后续对话内容始终共用该kv cache。它会将模型分为三个阶段：prompt推理、prefill推理、decode推理。
如果prompt不变，则prompt推理只用进行一次。
方法：在`llm_converter.py`命令加入`--share_prompt`，并指定`--max_prefill_kv_length`

可以直接用以下编译好的模型验证:
``` shell
# 8K上下文，prompt最大长度为4K，每轮对话输入最大长度是512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-4b-awq_w4f16_seq8192_bm1684x_1dev_20250825_144534.bmodel
```

## 模型编译

``` shell
# -s 指定总长度; --max_input_length 指定每次输入的最大长度; --max_prefill_kv_length 指定prompt共享的最大长度，会生成共用的kv cache传递给prefill
llm_convert.py -m /workspace/Qwen3-4B-AWQ -s 8192 --quantize w4f16 -c bm1684x --share_prompt --max_input_length 512 --max_prefill_kv_length 4096 --out_dir qwen3_4b_share
```

## 运行
```shell
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
python3 pipeline.py -m ./qwen3-4bxxxx.bmodel -c ../config --prompt test.txt
```
