# Command

## Modify sequence length

After switching to the Qwen/compile/ directory, manually change `"seq_length": 512,` in files/Qwen-7B-Chat/config.json to the length you need.
```shell
cd /workspace/LLM-TPU/models/Qwen/compile
vi files/Qwen-7B-Chat/config.json
```

PS:
1. Since a static ONNX model is exported, you must manually modify it to the length you need.

## Export onnx

```shell
cp files/Qwen-7B-Chat/* your_torch_path
export PYTHONPATH=your_torch_path:$PYTHONPATH
pip install transformers_stream_generator einops tiktoken accelerate transformers==4.32.0
```

### export basic onnx
```shell
python export_onnx.py --model_path your_torch_path --device cuda --seq_length 8192
```

### export onnx used for parallel demo
```shell
python export_onnx.py --model_path your_torch_path --device cuda --lmhead_with_topk 1
```

PS:
1. It is recommended to export with CUDA; when exporting blocks on CPU, the process gets stuck at the first block and can only be killed.
2. your_torch_path: the path of the model downloaded from the official website or trained by yourself, e.g. ./Qwen-7B-Chat
3. For long answers, you can use export_onnx_jacobi.py to export for acceleration (refs: https://github.com/hao-ai-lab/LookaheadDecoding)

## Compile bmodel

```shell
pushd /path_to/tpu-mlir
source envsetup.sh
popd
```

### compile basic bmodel
```shell
./compile.sh --mode int4 --name qwen-7b --addr_mode io_alone --seq_length 8192
```

### compile jacobi bmodel
```shell
./compile_jacobi.sh --mode int4 --name qwen-7b --addr_mode io_alone --generation_mode sample --decode_mode jacobi --seq_length 8192
```

### compile bmodel for parallel demo
```shell
./compile.sh --mode int4 --name qwen-7b --addr_mode io_alone --seq_length 8192 --num_device 8
```

PS:
1. mode: quantization method; currently supports fp16/bf16/int8/int4
2. name: model name; the Qwen series currently supports Qwen-1.8B/Qwen-7B/Qwen-14B
3. addr_mode: address allocation mode; you can use io_alone for acceleration
4. generation_mode: token sampling mode; when empty, greedy search is used; when set to sample, topk+topp is used; when set to all, topk + topp + temperature + repeat_penalty + max_new_tokens are used and can be passed in as parameters
5. decode_mode: decoding mode; when empty, normal decoding is used; when set to jacobi, jacobi decoding is used, which is only meaningful if export_onnx_jacobi.py was used for export earlier
6. seq_length: the maximum token length supported by the model

## Run Demo

For PCIE, it is recommended to create a new docker container, separate from the docker used to compile the bmodel, to clear the environment; otherwise errors may occur.
```
docker run --privileged --name your_docker_name -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash
docker exec -it your_docker_name bash
```

### python demo

For the python demo, be sure to source envsetup.sh inside LLM-TPU (it is different from the envsetup.sh in tpu-mlir).
```shell
cd /workspace/LLM-TPU
source envsetup.sh
```

```
cd /workspace/LLM-TPU/models/Qwen/python_demo
mkdir build && cd build
cmake .. && make
cp *cpython* ..
cd ..


python3 pipeline.py --model_path qwen-7b_int4_1dev.bmodel --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy
```

### cpp demo
```shell
cd /workspace/LLM-TPU/models/Qwen/demo
mkdir build && cd build
cmake .. && make
cp qwen_jacobi ..
cd ..
```

jacobi
```
./qwen_jacobi --model ../compile/qwen-7b_int4_1dev.bmodel --tokenizer ../support/qwen.tiktoken --devid 0
```

topk + topp + temperature + max_new_tokens + repeat_penalty
```
./qwen --model ../compile/qwen-7b_int4_1dev.bmodel --tokenizer ../support/qwen.tiktoken --devid 10 --top_p 0.8 --repeat_penalty 1.1 --repeat_last_n 32 --generation_mode sample --input_mode prompted
```

PS:
1. Based on current tests, the python demo and the cpp demo are basically the same in speed.
2. generation_mode: when set to basic, it means the model exported by export_onnx.py only uses a greedy LmHead; when set to greedy, it means the model exported by export_onnx.py supports sampling but greedy search is used; when set to sample, it means the model exported by export_onnx.py supports sampling and sampling is used.
3. input_mode: when set to prompted, prompts are applied automatically; when set to unprompted, the raw input is used without prompts (only effective when generation != basic).
