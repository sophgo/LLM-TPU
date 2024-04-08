```
cp ../compile/files/Qwen1.5-1.8B-Chat/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/

python export_onnx.py -m ../compile/Qwen1.5-1.8B-Chat/ -d cpu
```

```
python export_pt.py -m ../compile/Qwen1.5-7B-Chat/ -d cpu
./compile_pt.sh --mode int4 --name qwen1.5-7b --addr_mode io_alone --guess_len 5 --seq_length 512
```


```
mkdir build
cd build/ && cmake .. && make -j  && cp *cpython* .. && cd ..

python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-1.8b_int4_1dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-1.8b_int4_1dev.bmodel


```

```
draft_model = qwen1.5_0.5b_int4
target_model = qwen1.5_7b_int4
测试完成481个token用时40.94239926338196 s，提升20%

完成26个token用时2.04833984375 s，提升30%
```