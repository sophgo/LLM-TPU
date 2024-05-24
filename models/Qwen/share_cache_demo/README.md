```
cp files/Qwen-7B-Chat/* ../prompt_cache_demo/Qwen-7B-Chat/
python export_onnx.py --model_path ../prompt_cache_demo/Qwen-7B-Chat --device cpu --seq_length 8192 --num_thread 16 --batch_size 2 --share_length 6144
./compile.sh --mode int4 --name qwen-7b --share_length 6144 --unshare_length 1024 --addr_mode io_alone
```

编译库文件
```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

# python demo
```
python3 pipeline.py --model_path qwen-7b_int4_shareseq6144_1dev.bmodel --tokenizer_path ../support/token_config/ --devid 1 --generation_mode greedy
```
