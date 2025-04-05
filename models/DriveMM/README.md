# 目录说明
```
.
├── README.md
├── compile
│   ├── compile.sh                          #用来编译TPU模型的脚本
│   ├── export_onnx.py                      #用来导出onnx的脚本
│   └── files                               #用于替换原模型的文件
├── python_demo
│   ├── chat.cpp                            #主程序文件
└── └──pipeline.py                         #python demo的执行脚本
```
----------------------------

# 编译与运行
如果你不想从头编译模型，前三步可以直接省略，直接进入第四步

### 一：环境安装

```bash
sudo apt-get update
pip3 install transformers==4.45.1
```

### 二：生成onnx

```bash
cd compile
cp files/DriveMM/modeling_llama.py /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
python export_onnx.py
```

### 三：生成bmodel

生成2048长度默写
```bash
./compile.sh --seq_length 2048 --name drivemm
```

### 四：运行模型
[python_demo](./python_demo/README.md)

## 模型推理(Python Demo版本)
参考

## 性能测试

|   测试平台   |           测试模型              | 量化方式 | 模型长度 | first token latency(s) | token per second(tokens/s) |
| ----------- | ------------------------------ | -------- | -------- | --------------------- | -------------------------- |
| SE7-32      | drivemm                        | INT4     | 2048     | 3.484                 | 8.247                      |