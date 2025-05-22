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


# 运行
``` shell
# 下载编译好的模型: InternVL3-8b / InternVL3-2b
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/internVL3-8b_w4bf16_seq4096_bm1684x.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/internVL3-2b_w4bf16_seq4096_bm1684x.bmodel


cd python_demo
mkdir build && cd build 
cmake .. && make && mv chat.*so ..

python pipeline.py -m $bmodel_path -d $device_id
```
