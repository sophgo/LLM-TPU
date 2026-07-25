# Command

## Export onnx

```shell
pip install transformers_stream_generator einops tiktoken accelerate transformers==4.32.0
```

### export onnx
../compile/Qwen-7B-Chat/ is the location of your model
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

### Download the migrated model
You can also directly download the pre-compiled model instead of compiling it yourself
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen-7b_int4_1dev_8192_slide.bmodel
```

### python demo

For the python demo, you must source envsetup.sh inside LLM-TPU (which is different from the envsetup.sh inside tpu-mlir)
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
