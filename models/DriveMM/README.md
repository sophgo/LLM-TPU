# Directory description
```
.
├── README.md
├── compile
│   ├── compile.sh                          #Script used to compile the TPU model
│   ├── export_onnx.py                      #Script used to export onnx
│   └── files                               #Files used to replace the original model
├── python_demo
│   ├── chat.cpp                            #Main program file
└── └──pipeline.py                         #Execution script of the python demo
```
----------------------------

# Compile and run
If you do not want to compile the model from scratch, you can skip the first three steps and go directly to step 4

### 1: Environment installation

```bash
sudo apt-get update
pip3 install transformers==4.45.1
```

### 2: Generate onnx

```bash
cd compile
cp files/DriveMM/modeling_llama.py /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
python export_onnx.py
```

### 3: Generate bmodel

Generate a 2048-length model
```bash
./compile.sh --seq_length 2048 --name drivemm
```

### 4: Run the model
[python_demo](./python_demo/README.md)

## Model inference (Python Demo version)
Reference

## Performance test

|   Test platform   |           Test model              | Quantization method | Model length | first token latency(s) | token per second(tokens/s) |
| ----------- | ------------------------------ | -------- | -------- | --------------------- | -------------------------- |
| SE7-32      | drivemm                        | INT4     | 2048     | 3.484                 | 8.247                      |
