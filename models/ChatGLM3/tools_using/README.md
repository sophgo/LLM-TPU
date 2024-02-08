# ChatGLM3-BM1684X 工具调用
本文档将介绍在BM1684X上面运行的ChatGLM3如何进行工具调用。
大部分注意事项可以参考ChatGLM3: [工具调用](https://github.com/THUDM/ChatGLM3/blob/main/tool_using/README.md)


## 编译ChatGLM3-BM1684X Runtime
### 环境要求
- 支持C++17标准的```gcc```/```clang```
- ```cmake```版本不低于3.24
- python 版本不低于3.8, libpython3-dev, python3-numpy, swig
- 网络环境 需要从github下载部分依赖

可以参考我们的编译环境：
```docker
FROM ubuntu:22.04 AS build
RUN apt-get update && apt-get install -y gcc g++ cmake make git python3 libpython3-dev swig python3-numpy
```
### 编译
``` shell
    cd tools_using/src/
    mkdir build && cd build
    cmake .. -GNinja
    ninja
```
编译完成后，将```python/pyglm2.py```和```python/_pyglm2.so```复制到```tool_using```目录下:
``` shell
    cp python/pyglm2.py ../../
    cp python/_pyglm2.so ../../
```
## 运行ChatGLM3-BM1684X-Cli-Demo
将 ```chatglm3-6b_int8.bmodel```, ```tokenizer.model```, ```chatglm3-6b```都放到同一个目录下。
示例：

```
├── chatglm3-6b
│   ├── config.json
│   ├── configuration_chatglm.py
│   ├── modeling_chatglm.py
│   ├── MODEL_LICENSE
│   ├── pytorch_model-00001-of-00007.bin
│   ├── pytorch_model-00002-of-00007.bin
│   ├── pytorch_model-00003-of-00007.bin
│   ├── pytorch_model-00004-of-00007.bin
│   ├── pytorch_model-00005-of-00007.bin
│   ├── pytorch_model-00006-of-00007.bin
│   ├── pytorch_model-00007-of-00007.bin
│   ├── pytorch_model.bin.index.json
│   ├── quantization.py
│   ├── README.md
│   ├── tokenization_chatglm.py
│   ├── tokenizer_config.json
│   └── tokenizer.model
├── chatglm3-6b_int8.bmodel
└── tokenizer.model

```
执行

``` shell
    python cli_demo_tool_tpu.py [model_directory]
```
根据提示输入指令即可，在当前demo中的示例tool是查询股票价格：
``` python
tools = [{'name': 'track', 'description': '追踪指定股票的实时价格', 'parameters': {
    'type': 'object', 'properties': {'symbol': {'description': '需要追踪的股票代码'}}, 'required': []}}]
```

### 示例

**用户：** 查询英伟达股价

**ChatGLM3-TPU:** `{'name': 'track', 'parameters': {'symbol': 'NVIDIA'}}`

**结果：** `{"price": 487}`


<sub><sup>*这里需要自行实现调用工具的逻辑。假设已经得到了返回结果，将结果以 json 格式返回给模型并得到回复。*</sup></sub>

**ChatGLM3-TPU:** 根据您的查询，我已经调用了追踪股票实时价格的API，查询到英伟达（NVIDIA）的当前股价为487美元。






