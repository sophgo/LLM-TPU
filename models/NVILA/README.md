# 编译

``` shell
cd compile

llm_convert.py -m $model_path -s $seq_length -q w4f16

python export_onnx.py -m $model_path -i $image_path

./compile.sh
```
* llm_convert.py暂时不支持vit部分直接导出

先通过llm_convert.py直接编译llm部分bmodel，再通过export_onnx.py导出vit部分的onnx，最后通过compile.sh编译vit部分的bmodel，并combine到一起。


export_onnx.py 添加-i参数指定image, 可以编译该image shape下的静态bmodel, 可以提升相同shape下的推理性能


# 运行
``` shell
# 下载编译好的模型
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/nvila-8b_seq4096_w4f16_bm1684x.bmodel


cd python_demo
mkdir build && cd build 
cmake .. && make && mv chat.*so ..

python pipeline.py -m $bmodel_path -d $device_id
```
