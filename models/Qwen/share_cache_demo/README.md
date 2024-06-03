# Command
your_torch_model是你模型的路径
```
pip install transformers_stream_generator einops tiktoken accelerate transformers==4.32.0

cp files/Qwen-7B-Chat/* your_torch_model

./compile.sh --mode int4 --name qwen-7b --share_length 6016 --addr_mode io_alone --unshare_length 1536 --dynamic 1
```

# 直接下载
如果你不想编译模型，也可以直接下载
```
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen-7b_int4_shareseq6016_unshare1536_seq7552_1dev_dyn.bmodel
```
* 使用的TPU-MLIR版本： bacc66292743153ff2f16927bffee69ffacb476c
* 内存：9663MB（动态）

# 分片方式
|第一片                  |第二片                 |第三片              |
|:-                     |:-                     |:-                 |
|share                  |unshare                |decode             |
|share_length=6016      |unshare_length=1536    |decode_length=0    |

# 编译库文件
```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

# python demo
```
python3 pipeline.py --model_path qwen-7b_int4_shareseq6016_unshare1536_seq7552_1dev.bmodel --tokenizer_path ../support/token_config/ --devid 0 --generation_mode penalty_sample
```

# 效果图
![](../../../assets/Qwen_share_cache_demo.png)
