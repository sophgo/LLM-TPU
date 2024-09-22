# 多芯demo

根据文档安装好驱动和docker环境，以下操作都需要在docker中进行

## 1. 编译

按如下操作编译glm4-9b模型：
```shell
# 先将提供的模型配置文件替换至下载的权重内
export ChatGLM4_PATH=$PWD/glm-4-9b-chat
cd GLM4/compile/
pushd files/glm-4-9b-chat/
cp ./compile/files/glm-4-9b-chat/modeling_chatglm.py $ChatGLM4_PATH
cp ./compile/files/glm-4-9b-chat/config.json $ChatGLM4_PATH
popd

# 根据需要的sequence length导出onnx，注意多芯必须添加lmhead_with_topk参数
python3 ./export_onnx.py -m $ChatGLM4_PATH -s 2048 --lmhead_with_topk

# 静态编译
./compile.sh --mode int4 --num_device 8 --name glm4-9b --seq_length 2048
```

如果不打算编译模型，可以通过以下命令下载已编译好的模型，目前有如下模型已经预编译好，**注意最新版本的驱动需要重新下载下方的模型**：
```shell
pip3 install dfss
# int4
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/glm4_seq2048_w4f16_8dev.bmodel
# int8
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/glm4_seq2048_w8f16_8dev.bmodel
```

## 3. 运行
```shell
git submodule update --init

cd python_demo_parallel
mkdir build 
cd build && cmake .. && make -j8 && cp *cpython* .. && cd ..
python3 pipeline.py --model_path glm4_seq2048_w8f16_8dev.bmodel --tokenizer_path ../token_config/ --devid 0,1,2,3,4,5,6,7
```

运行web demo
```shell
pip3 install gradio==3.39.0 mdtex2html==1.2.0 dfss
python3 web_demo.py --model_path ./glm4_seq2048_w8f16_8dev.bmodel --tokenizer_path ..token_config/ --devid 0,1,2,3,4,5,6,7
```
