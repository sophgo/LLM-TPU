推理命令在run.sh里面

```bash
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```
记得替换下面的bmodel地址

```bash
cd models/RWKV6/python_demo/
python pipeline.py --model_path /data/work/LLM-TPU-RWKV-dev/bmodels/rwkv6-1b5_bf16_1dev.bmodel --tokenizer_path ./rwkv_vocab_v20230424.txt --devid 0 --generation_mode greedy
```
