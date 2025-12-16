## Qwen2-Audio
本工程实现BM1684X上的Qwen2-Audio部署多模态大模型Qwen2-Audio。通过TPU-MLIR将模型转换为BM1684X上的bmodel，实现高效的推理。并采用c++代码将其部署到BM1684X上，提供python接口调用。目前仅实现了SoC环境。

## 开发环境准备
从Huggingface或者modelscope上下载Qwen2-Audio模型Qwen2-Audio-7B-Instruct文件。

## 编译模型

此处介绍如何将onnx模型编译成bmodel。可以直接下载编译好的模型。
```
# 1684X
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2-audio-7b_w8f16_seq599_1dev.bmodel

```
### 下载docker，启动容器
```
docker pull sophgo/tpuc_dev:latest
docker run -it --rm --privileged --net=host --ipc=host -v $(pwd):/workspace sophgo/tpuc_dev:latest
docker exec -it $(docker ps -lq) /bin/bash
```
### 下载TPU-MLIR代码并编译
```
git clone https://github.com/sophgo/tpu-mlir.git
cd tpu-mlir
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DLLVM_TARGETS_TO_BUILD="BPF;X86" -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
### 导出onnx模型
将tools/文件里面的model_qwen2.py替换transformers里面的有关qwen2的相关文件。然后使用运行：
```
python3 export_onnx.py
```
导出模型，有些模型不适合导出onnx，仅需要导出pt文件，详情件export_onnx.py文件。
### 编译生成bmodel
将模型放置在合适的目录下，然后运行compile.sh脚本。
```
./compile.sh
```
## 编译与运行程序
编译库文件，生成chat.cpython*.so文件，将该文件拷贝到pipeline.py文件目录; 并将bmodel文件和config目录拷贝过去
```
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
python demo
python3 pipeline.py -m qwen2-audio-7b_w8f16_seq599_1dev.bmodel -c config
```

### 评价指标

| Metric | Dataset-Split                     | qwen2-audio-chat(fp32,A6000) | qwen2-audio-chat(fp16,A6000) | qwen2-audio-chat(bf16,A6000) | qwen2-audio-chat(w4f16, airbox) | qwen2-audio-chat(w8f16,airbox) |
|--------|-----------------------------------|-----------------------------|-------------------------------|------------------------------|---------------------------------|--------------------------------|
| WER⬇   | librispeech-test-clean (2619个)   | 2.33                   | 2.32                   | 2.33                   | 14.98                    | 19.10                    |

### 评价工具
https://github.com/OpenBMB/UltraEval-Audio
