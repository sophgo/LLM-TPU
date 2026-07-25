# RWKV7

This project deploys the language large model [RWKV7](https://modelscope.cn/models/Blink_DL/rwkv-7-world) on BM1684X. The model is converted into a bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed to the BM1684X PCIE or SoC environment using C++ code.


RWKV is an RNN-structured network with GPT-level large language model performance, combining the best features of RNNs and Transformers: excellent performance, constant memory usage, constant inference generation speed, "infinite" ctxlen, and free sentence embeddings, all with 100% no self-attention mechanism. For detailed features and usage examples, please refer to the official documentation [RWKV7](https://www.rwkv.cn/)

# Directory structure
```
.
├── README.md
├── compile
│   ├── compile.sh                          # script for compiling the bmodel
│   ├── export_onnx.py                      # script for exporting onnx
├── python_demo
│   ├── chat.cpp                            # inference script
│   └── pipeline.py                         # python execution script
└── tokenizer                               # tokenizer
    ├── rwkv_tokenizer.py
    └── rwkv_vocab_v20230424.txt
```
----------------------------

# Model compilation

* Model compilation must be done inside docker; it cannot be done outside docker
* If you do not want to compile the model, you can also jump directly to [Model deployment](#model-deployment) to test our precompiled model.

The model compilation process converts the original weights to onnx, then converts the model into a bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler. For MLIR environment setup, please refer to: [MLIR Environment Installation Guide](https://github.com/sophgo/LLM-TPU/blob/main/docs/Mlir_Install_Guide.md)

## 1. Generate the onnx file

``` shell
cd compile
python export_onnx.py -m your_model_path
```
* your_model_path refers to the path where the original model was downloaded, e.g. "rwkv-7-world/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth".

rwkv only requires torch and numpy dependencies. The following parameters are also available for testing CPU results:
| **Option**             | **Takes an argument** | **Default value** | **Description**                                             |
|------------------------|------------------|----------------|-----------------------------------------------------------|
| `-c`, `--chunk_len`    | Yes              | 32              | The number of chunks used by rwkv during prefill; only affects inference speed |
| `-t`, `--test`         | No               | None            | Test rwkv7 CPU inference                                  |
| `-s`, `--state_path`   | Yes              | None            | The rwkv state file; a scenario-specific state can be loaded in advance for testing CPU inference |

## 2. Generate the bmodel file

``` shell
./compile.sh
```
* Generating the rwkv 0.1B bmodel takes more than about 2 hours. One bmodel is generated for prefill and one for decode; the long wait in between is normal and does not mean compilation is stuck
* rwkv currently does not support w4/w8 precision; the default is f16

----------------------------

# Model deployment

* If you do not want to compile the model, you can directly download the precompiled model:
```shell
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/rwkv7-0.1b_chunk64_f16.bmodel
```

## Compile the program
Run the following compilation (the PCIE version is the same as the SoC version):

```shell
cd python_demo
mkdir build && cd build && cmake .. && make && cp *chat* ..
cd ..
```

## Model inference
```shell
python3 pipeline.py -m bmodel_path -d your_devid
```
Other available parameters can be viewed in `pipeline.py` or by running the following command
```shell
python3 pipeline.py --help
```
For rwkv decoding parameters, please refer to [RWKV decoding parameters](https://www.rwkv.cn/docs/RWKV-Prompts/RWKV-Parameters)
For the rwkv state file, please refer to [RWKV state loading](https://rwkv.cn/news/read?id=343)