# Command

## Export onnx

```shell
cp files/Phi-3-mini-128k-instruct/modeling_phi3.py your_torch_path
export PYTHONPATH=your_torch_path:$PYTHONPATH
pip install transformers_stream_generator einops tiktoken accelerate transformers==4.37.2
```

### export basic onnx
```shell
python export_onnx.py --model_path your_torch_path --seq_length your_length
```

PS：
1. your_torch_path：从官网下载的或者自己训练的模型的路径，例如./InternVL-4B/

## Compile bmodel

```shell
pushd /path_to/tpu-mlir
source envsetup.sh
popd
```

### Compile LLM-Vision bmodel
```shell
./compile.sh --mode int4 --name phi3-4b
```

PS：
1. mode：量化方式，目前支持fp16/int8/int4
2. name：模型名称，目前phi3系列支持 phi3-4b
3. addr_mode：地址分配方式，可以使用io_alone方式来加速
4. seq_length：模型支持的最大token长度

## Run Demo

如果是pcie，建议新建一个docker，与编译bmodel的docker分离，以清除一下环境，不然可能会报错
```
docker run --privileged --name your_docker_name -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash
docker exec -it your_docker_name bash
```

