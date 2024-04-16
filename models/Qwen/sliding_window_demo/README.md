# Command

## Export onnx

```shell
pip install transformers_stream_generator einops tiktoken accelerate transformers==4.32.0
```

### export onnx
../compile/Qwen-7B-Chat/是你模型的位置
```shell
cp ../compile/files/Qwen-7B-Chat/modeling_qwen.py ../compile/Qwen-7B-Chat/modeling_qwen.py

python export_onnx.py --model_path ../compile/Qwen-7B-Chat/ --device cpu --seq_length 8192 --window_length 1024
```

## Compile bmodel

```shell
pushd /path_to/tpu-mlir
source envsetup.sh
popd
```

### compile bmodel
```shell
./compile.sh --mode int4 --name qwen-7b --seq_length 8192
```

### 下载迁移好的模型
也可以直接下载编译好的模型，不用自己编译
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen-7b_int4_1dev_8192_slide.bmodel
```

### python demo

对于python demo，一定要在LLM-TPU里面source envsetup.sh（与tpu-mlir里面的envsetup.sh有区别）
```shell
cd /workspace/LLM-TPU
source envsetup.sh
```

```shell
cd /workspace/LLM-TPU/models/Qwen/python_demo
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..

python3 pipeline.py --model_path qwen-7b_int4_1dev_8192_slide.bmodel --tokenizer_path ../token_config/ --devid 0 --generation_mode penalty_sample
```