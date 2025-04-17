# RWKV7

本项目实现BM1684X部署语言大模型[RWKV7](https://modelscope.cn/models/Blink_DL/rwkv-7-world)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。


RWKV 是一种具有 GPT 级大型语言模型性能的 RNN 结构网络, 结合了 RNN 和 Transformer 的最佳特性：出色的性能、恒定的显存占用、恒定的推理生成速度、"无限" ctxlen 和免费的句嵌入，而且 100% 不含自注意力机制。具体特性以及使用示例，可以参考官方文档[RWKV7](https://www.rwkv.cn/)

# 目录说明
```
.
├── README.md
├── compile
│   ├── compile.sh                          #用来编译bmodel的脚本
│   ├── export_onnx.py                      #用来导出onnx的脚本
├── python_demo
│   ├── chat.cpp                            #推理脚本
│   └── pipeline.py                         #python执行脚本
└── tokenizer                               #分词器
    ├── rwkv_tokenizer.py
    └── rwkv_vocab_v20230424.txt
```
----------------------------

# 模型编译

* 模型编译必须要在docker内完成，无法在docker外操作
* 如果不想编译模型，也可以直接跳转至[模型部署](#模型部署)测试我们编译好的模型。

模型编译的流程是将原始权重转换为onnx，再通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，MLIR具体环境配置请参考：[MLIR环境安装指南](https://github.com/sophgo/LLM-TPU/blob/main/docs/Mlir_Install_Guide.md)

## 1.生成onnx文件

``` shell
cd compile
python export_onnx.py -m your_model_path
```
* your_model_path 指的是原模型下载后的地址,比如 "rwkv-7-world/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth"。

rwkv只需要torch、numpy依赖，还有以下参数，可以测试cpu的结果：
| **选项**               | **是否需要参数** | **默认值**       | **描述**                                                  |
|------------------------|------------------|----------------|-----------------------------------------------------------|
| `-c`, `--chunk_len`    | 是               | 32              | rwkv在prefill时的分块数量，只影响推理速度                   |
| `-t`, `--test`         | 否               | 无              | 测试cpu推理rwkv7                                          |
| `-s`, `--state_path`   | 是               | None            | rwkv的state文件，可以提前加载特定场景的state，用于测试cpu推理   |

## 2.生成bmodel文件

``` shell
./compile.sh
```
* 生成rwkv 0.1B bmodel耗时大概2小时以上，prefill/decode各一个bmodel，中间耗时长，并不是编译卡住
* rwkv目前不支持w4/w8精度，默认f16

----------------------------

# 模型部署

* 如果不想编译模型可以直接下载编译好的模型：
```shell
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/rwkv7-0.1b_chunk64_f16.bmodel
```

## 编译程序
执行如下编译，(PCIE版本与SoC版本相同)：

```shell
cd python_demo
mkdir build && cd build && cmake .. && make && cp *chat* ..
cd ..
```

## 模型推理
```shell
python3 pipeline.py -m bmodel_path -d your_devid
```
其它可用参数可以通过`pipeline.py` 或者执行如下命令进行查看 
```shell
python3 pipeline.py --help
```
rwkv的解码参数可以参考[RWKV的解码参数](https://www.rwkv.cn/docs/RWKV-Prompts/RWKV-Parameters)
rwkv的state文件可以参考[RWKV的state加载](https://rwkv.cn/news/read?id=343)