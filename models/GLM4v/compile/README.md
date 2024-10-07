# Command

## Modify sequence length

切换到GLM4v/compile/目录下后，手动将files/glm-4v-9b/config.json中的`"seq_length": 8196,`修改为你需要的长度
注意！GLM4v中图片固定占1600个token，所以修改长度时需要将图片长度也考虑进去，seq_length必须大于1600
```shell
cd /workspace/LLM-TPU/models/GLM4v/compile
vi files/config.json
```

PS：
1. 由于导出的是静态onnx模型，所以必须手动修改为你所需要的长度

## Export onnx

```shell
cp files/Qwen-VL-Chat/* your_torch_path
export PYTHONPATH=your_torch_path:$PYTHONPATH
pip install transformers_stream_generator einops tiktoken accelerate transformers==4.44.0
```

### export basic onnx
```shell
python export_onnx.py --model_path your_torch_path --device cuda
```

### export onnx used for parallel demo
```shell
python export_onnx.py --model_path your_torch_path --device cuda --lmhead_with_topk 1
```

PS：
1. 最好使用cuda导出，cpu导出block的时候，会卡在第一个block，只能kill
2. your_torch_path：从官网下载的或者自己训练的模型的路径，例如./glm-4v-9b

## Compile bmodel

```shell
pushd /path_to/tpu-mlir
source envsetup.sh
popd
```

### compile basic bmodel
```shell
./compile.sh --mode int4 --name glm4v-9b --addr_mode io_alone --seq_length 1024
```


PS：
1. mode：量化方式，目前支持fp16/bf16/int8/int4
2. name：模型名称，目前Qwen系列支持 Qwen-1.8B/Qwen-7B/Qwen-14B/Qwen-VL-Chat
3. addr_mode：地址分配方式，可以使用io_alone方式来加速
4. generation_mode：token采样模式，为空时，使用greedy search；为sample时，使用topk+topp；为all时使用topk + topp + temperature + repeat_penalty + max_new_tokens，并且可以作为参数传入
5. decode_mode：编码模式，为空时，使用正常编码；为jacobi时，使用jacobi编码，只有前面使用export_onnx_jacobi.py时，用jacobi才有意义
6. seq_length：模型支持的最大token长度



### 改动位置

源代码位置557行~564行
```
if kv_cache is not None:
    cache_k, cache_v = kv_cache
    key_layer = torch.cat((cache_k, key_layer), dim=2)
    value_layer = torch.cat((cache_v, value_layer), dim=2)
if use_cache:
    kv_cache = (key_layer, value_layer)
else:
    kv_cache = None
```
修改为
```
if use_cache:
    kv_cache = (key_layer, value_layer)
else:
    kv_cache = None
if kv_cache is not None:
    cache_k, cache_v = kv_cache
    key_layer = torch.cat((cache_k, key_layer), dim=2)
    value_layer = torch.cat((cache_v, value_layer), dim=2)

```
调整顺序，使得返回的kv_cache长度为1