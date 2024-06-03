# 编译模型
your_torch_model是你的torch模型，--dynamic 1是指prefill使用动态
```
pip3 install transformers==4.37.0

cp files/Qwen1.5-4B-Chat/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/

python export_onnx.py --model_path ../compile/Qwen1.5-4B-Chat --device cpu --share_length 6144 --unshare_length 2560 --seq_length 8704 --num_thread 16

./compile.sh --mode int4 --name qwen1.5-4b --share_length 6144 --addr_mode io_alone --unshare_length 2560 --dynamic 1
```

# 直接下载
如果你不想编译模型，也可以直接下载
```
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-4b_int4_shareseq6144_unshare2560_seq8704_1dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-4b_int4_shareseq6144_unshare2560_seq8704_1dev_dyn.bmodel
```
* 使用的TPU-MLIR版本： bacc66292743153ff2f16927bffee69ffacb476c
* 运行时内存：6958MB（动态）

# 分片方式
|第一片                  |第二片                 |第三片              |
|:-                     |:-                     |:-                 |
|share                  |unshare                |decode             |
|share_length=6144      |unshare_length=2560    |decode_length=0    |

# 编译库文件
```
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

# python demo
```
python3 pipeline.py --model_path qwen1.5-4b_int4_shareseq6144_unshare2560_seq8704_1dev_dyn.bmodel --tokenizer_path ../token_config/ --devid 0 --generation_mode penalty_sample
```

# 效果图
![](../../../assets/Qwen1_5_share_cache_demo.png)
