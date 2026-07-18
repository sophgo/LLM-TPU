## Qwen2-Audio
This project implements the deployment of the multimodal large model Qwen2-Audio on BM1684X. The model is converted to bmodel on BM1684X through TPU-MLIR to achieve efficient inference. It is deployed to BM1684X using c++ code, with a python interface provided for invocation. Currently only the SoC environment is implemented.

## Development environment preparation
Download the Qwen2-Audio model Qwen2-Audio-7B-Instruct files from HuggingFace or ModelScope.

## Compile the model

This section describes how to compile the onnx model into bmodel. You can directly download the pre-compiled model.
```
# 1684X
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2-audio-7b_w8f16_seq599_1dev.bmodel

```
### Download docker and start the container
```
docker pull sophgo/tpuc_dev:latest
docker run -it --rm --privileged --net=host --ipc=host -v $(pwd):/workspace sophgo/tpuc_dev:latest
docker exec -it $(docker ps -lq) /bin/bash
```
### Download TPU-MLIR code and compile
```
git clone https://github.com/sophgo/tpu-mlir.git
cd tpu-mlir
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DLLVM_TARGETS_TO_BUILD="BPF;X86" -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```
### Export the onnx model
Replace the qwen2-related files in transformers with the model_qwen2.py in the tools/ directory. Then run:
```
python3 export_onnx.py
```
to export the model. Some models are not suitable for exporting to onnx and only need to be exported as pt files. See the export_onnx.py file for details.
### Compile to generate bmodel
Place the model in an appropriate directory, then run the compile.sh script.
```
./compile.sh
```
## Compile and run the program
Compile the library file to generate the chat.cpython*.so file, and copy this file to the directory where pipeline.py is located; also copy the bmodel file and the config directory over
```
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
python demo
python3 pipeline.py -m qwen2-audio-7b_w8f16_seq599_1dev.bmodel -c config
```
