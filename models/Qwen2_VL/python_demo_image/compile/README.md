# Command

## Export onnx

```shell
pip install qwen-vl-utils accelerate torch==2.5.0 transformers==4.45.1
pip show transformers
cp files/Qwen2-VL-2B-Instruct/modeling_qwen2_vl.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py
```
* 不一定是`/usr/local/lib/python3.10/dist-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py`这个路径，可以用pip show transformers查看


### export onnx
```shell
python export_onnx.py --model_path your_torch_path --seq_length 512
```
PS：
1. your_torch_path：从官网下载的或者自己训练的模型的路径，例如./Qwen2-VL-7B-Instruct

## Compile bmodel

```shell
pushd /path_to/tpu-mlir
source envsetup.sh
popd
```

### compile basic bmodel
```shell
./compile.sh --mode int4 --name qwen2-vl --addr_mode io_alone --seq_length 512
```

## Run Demo

### python demo

```shell
cd ../python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..

python3 pipeline.py --model_path qwen2-vl-2b_int4_seq512_1dev.bmodel --tokenizer_path ../support/token_config/ --processor_path ../support/processor_config --devid 0 --generation_mode greedy
```