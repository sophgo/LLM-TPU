The inference command is in run.sh

```bash
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```
Remember to replace the bmodel path below

```bash
cd models/RWKV6/python_demo/
python pipeline.py --model_path /data/work/LLM-TPU-RWKV-dev/bmodels/rwkv6-1b5_bf16_1dev.bmodel --tokenizer_path ./rwkv_vocab_v20230424.txt --devid 0 --generation_mode greedy
```
