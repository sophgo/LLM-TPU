# Command

## Export onnx

```shell
pip install transformers_stream_generator einops tiktoken accelerate transformers==4.37.0
```

### export onnx
your_torch_model是你模型的位置
```shell
cp files/Qwen1.5-7B-Chat/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/
```

导出onnx
```shell
python3 export_onnx.py --model_path Qwen1.5-4B-Chat --seq_length your_sequence_length
```

## Compile bmodel

```shell
pushd /path_to/tpu-mlir
source envsetup.sh
popd
```

### compile bmodel
```shell
./compile.sh --mode int4 --name qwen1.5-4b --seq_length 8192
```
使用io_alone
```
./compile.sh --mode int4 --name qwen1.5-4b --addr_mode io_alone --seq_length 8192
```

### 下载迁移好的模型
也可以直接下载编译好的模型，不用自己编译
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-4b_int4_1dev_8k.bmodel
```

### python demo

请见python_demo里面的README
