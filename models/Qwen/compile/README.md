# Command

## Modify sequence length

切换到Qwen/compile/目录下后，手动将files/Qwen-7B-Chat/config.json中的`"seq_length": 512,`修改为你需要的长度
```shell
cd /workspace/LLM-TPU/models/Qwen/compile
vi files/Qwen-7B-Chat/config.json
```

PS：
1. 由于导出的是静态onnx模型，所以必须手动修改为你所需要的长度

## Export onnx

```shell
cp files/Qwen-7B-Chat/* your_torch_path
export PYTHONPATH=your_torch_path:$PYTHONPATH
pip install transformers_stream_generator einops tiktoken accelerate transformers=4.32.0
```

### export basic onnx
```shell
python export_onnx.py --model_path your_torch_path --generation_mode sample --device cuda
```

### export jacobi onnx
```shell
python export_onnx_jacobi.py --model_path your_torch_path --guess_len 8 --generation_mode sample --device cuda
```

PS：
1. 最好使用cuda导出，cpu导出block的时候，会卡在第一个block，只能kill
2. your_torch_path：从官网下载的或者自己训练的模型的路径，例如./Qwen-7B-Chat
3. generation_mode：生成token的方式，目前支持两种，greedy是贪婪采样，sample是使用topk+topp
4. 对于长回答，可以使用export_onnx_jacobi.py来导出加速（refs：https://github.com/hao-ai-lab/LookaheadDecoding）

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
./compile.sh --mode int4 --name qwen-7b --addr_mode io_alone --generation_mode sample --decode_mode jacobi --seq_length 8192
```

PS：
1. mode：量化方式，目前支持fp16/bf16/int8/int4
2. name：模型名称，目前Qwen系列支持 Qwen-1.8B/Qwen-7B/Qwen-14B
3. addr_mode：地址分配方式，可以使用io_alone方式来加速
4. generation_mode：token采样模式，为空时，使用greedy search，为sample，使用topk+topp
5. decode_mode：编码模式，为空时，使用正常编码，为jacobi时，使用jacobi编码，只有前面使用export_onnx_jacobi.py时，用jacobi才有意义
6. seq_length：模型支持的最大token长度

## Run Demo

如果是pcie，建议新建一个docker，与编译bmodel的docker分离，以清除一下环境，不然可能会报错
```
docker run --privileged --name your_docker_name -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash
docker exec -it your_docker_name bash
```

### python demo

对于python demo，一定要在LLM-TPU里面source envsetup.sh（与tpu-mlir里面的envsetup.sh有区别）
```shell
cd /workspace/LLM-TPU
source envsetup.sh
```

```
cd /workspace/LLM-TPU/models/Qwen/python_demo
mkdir build && cd build
cmake .. && make
cp chat_jacobi.cpython-310-x86_64-linux-gnu.so ..
cd ..


python chat.py --devid 0 --model_path ../compile/qwen-7b_int4_1dev.bmodel --tokenizer_path ../support/token_config/ --generation_mode sample --decode_mode jacobi
```

### cpp demo
```shell
cd /workspace/LLM-TPU/models/Qwen/demo
mkdir build && cd build
cmake .. && make
cp qwen_jacobi ..
cd ..

./qwen_jacobi --model ../compile/qwen-7b_int4_1dev.bmodel --tokenizer ../support/qwen.tiktoken --devid 3
```

PS：
1. 目前测试下来，python demo和cpp demo在速度上基本一致