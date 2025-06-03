# Moss

本工程实现BM1684X部署语言大模型[moss-moon-003-sft](https://huggingface.co/fnlp/moss-moon-003-sft)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并将其部署到BM1684X的PCIE环境。

## 开发环境准备

### 1.下载docker，启动容器

```
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```

后文假定环境都在docker的`/workspace`目录。

### 2.下载TPU-MLIR代码并编译

```
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  #激活环境变量
./build.sh #编译mlir
```

### 3. 导出onnx模型

参考[compile](./compile) 与 [compile](./compile) 下面的README.md

### 4. 编译模型

参考[compile](./compile) 与 [compile](./compile) 下面的README.md

### 5. 编译与运行程序

参考[python_demo](./python_demo) 下面的README.md

## 对modeling_moss.py文件代码做调整

### 第一处

原代码：

```python
mask_value = torch.finfo(attn_weights.dtype).min
mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
attn_weights = torch.where(causal_mask, attn_weights, mask_value)
```

修改后：注释掉这部分代码，因为可以直接采用`attention_mask`，避免复杂逻辑，提升性能

### 第二处

原代码：

```python
if layer_past is not None:
    past_key = layer_past[0]
    past_value = layer_past[1]
    key = torch.cat((past_key, key), dim=-2)
    value = torch.cat((past_value, value), dim=-2)
```

修改后：

```python
if layer_past is not None:
    past_key = layer_past[0]
    past_value = layer_past[1]
    past_key=past_key.permute(0, 2, 1, 3)
    past_value=past_value.permute(0, 2, 1, 3)
    key = torch.cat((past_key, key), dim=-2)
    value = torch.cat((past_value, value), dim=-2)
```

对齐past_key和past_value的形状