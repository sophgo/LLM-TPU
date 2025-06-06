# 编译

``` shell
cd compile
# 替换源码
cp files/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/

python export_onnx.py -m $model_path -s $seq_length

./compile.sh --name $model_name --seq_length $seq_length
```
编译长度为4k的 InternVL3-8B:

``` shell
python export_onnx.py -m $your_model_path/InternVL3-8B -s 4096
./compile.sh --name internvl3-8b --seq_length 4096

``` 

MLIR新增一键编译脚本：
``` shell
llm_convert.py -m /workspace/InternVL3-2B-AWQ -s 4096 -q w4bf16 -c bm1684x --out_dir internvl3-2b
```
编译完成后，在指定目录`internvl3-2b`生成`internvl3-2b-xxx.bmodel`和`config`，其中config包含tokenizer和其他原始config。添加--do_sample参数编译采样模型


也可以直接下载编译好的模型:
``` shell
# InternVL3-8b
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/internvl3-8b_w4bf16_seq4096_bm1684x.bmodel
# InternVL3-2b
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/internvl3-2b_w4bf16_seq4096_bm1684x.bmodel
```

# 运行
``` shell
cd python_demo
mkdir build && cd build 
cmake .. && make && mv chat.*so ..

python pipeline.py -m $bmodel_path -c $config_path -d $device_id
```
如果在编译时打开了--do_sample，运行时也可以选择加上--do_sample，根据config路径中的generation_config.json采样参数进行采样。

