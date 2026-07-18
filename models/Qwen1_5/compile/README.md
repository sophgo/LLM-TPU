# Command

## Export onnx

```shell
pip install transformers_stream_generator einops tiktoken accelerate transformers==4.37.0
cp files/Qwen1.5-1.8B-Chat/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/
```
your_torch_model is the location of your model
```shell
python3 export_onnx.py --model_path your_torch_model --seq_length 512
python3 export_onnx.py --model_path your_torch_model --seq_length 1280
```

## Compile bmodel
```shell
./compile.sh --mode int8 --name qwen1.5-1.8b --seq_length 512
./compile.sh --mode int8 --name qwen1.5-1.8b --seq_length 1280
```
Regarding io_alone, it is recommended to use io_alone when the sequence is relatively long; there is no need to use io_alone for lengths such as 512 or 1280
```
./compile.sh --mode int4 --name qwen1.5-4b --addr_mode io_alone --seq_length 8192
```

### Download the migrated model
You can also directly download the pre-compiled model instead of compiling it yourself
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-1.8b_int4_seq512_1dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-1.8b_int8_seq512_1dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-1.8b_int4_seq1280_1dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-1.8b_int8_seq1280_1dev.bmodel

python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-4b_int4_1dev_8k.bmodel
```

### python demo

Please see the README in python_demo
