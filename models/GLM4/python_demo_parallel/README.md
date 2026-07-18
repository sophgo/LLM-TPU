# Multi-chip demo

Install the driver and docker environment according to the documentation. All the following operations need to be performed inside docker

## 1. Compilation

Compile the glm4-9b model as follows:
Before compiling, you need to change the seq_length in config.json to the length you need

```shell
# If you are converting the official model, you can download the torch model from huggingface or from the following link
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/glm-4-9b-chat-torch.zip

# First, replace the provided model configuration files into the downloaded weights
export ChatGLM4_PATH=$PWD/glm-4-9b-chat
cd GLM4/compile/
pushd files/glm-4-9b-chat/
cp ./compile/files/glm-4-9b-chat/modeling_chatglm.py $ChatGLM4_PATH
cp ./compile/files/glm-4-9b-chat/config.json $ChatGLM4_PATH
popd

# Export onnx according to the required sequence length. Note that the lmhead_with_topk parameter must be added for multi-chip
python3 ./export_onnx.py -m $ChatGLM4_PATH -s 2048 --lmhead_with_topk 1

# Static compilation
./compile.sh --mode int4 --num_device 8 --name glm4-9b --seq_length 2048
```

If you do not plan to compile the model, you can download the pre-compiled model with the following commands. The following models are currently pre-compiled. **Note that the latest version of the driver requires re-downloading the models below**:
```shell
pip3 install dfss
# int4
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/glm4_seq2048_w4f16_8dev.bmodel
# int8
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/glm4_seq2048_w8f16_8dev.bmodel
```

## 3. Run
```shell
cd python_demo_parallel
mkdir build 
cd build && cmake .. && make -j8 && cp *cpython* .. && cd ..
python3 pipeline.py --model_path glm4_seq2048_w8f16_8dev.bmodel --tokenizer_path ../token_config/ --devid 0,1,2,3,4,5,6,7
```

Run the web demo
```shell
pip3 install gradio==3.39.0 mdtex2html==1.2.0 dfss
python3 web_demo.py --model_path ./glm4_seq2048_w8f16_8dev.bmodel --tokenizer_path ..token_config/ --devid 0,1,2,3,4,5,6,7
```
