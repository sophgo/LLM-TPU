# Qwen

本工程实现BM1684X部署语言大模型[Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

* 本工程也支持[Qwen-14-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat)，操作方法与`Qwen-7B-Chat`一致。
* 本工程也支持[Qwen-1_8-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat)，操作方法与`Qwen-7B-Chat`一致。

## 开发环境准备

### 1. 下载docker，启动容器

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```
后文假定环境都在docker的`/workspace`目录。

### 2. 下载`TPU-MLIR`代码并编译

(也可以直接下载编译好的release包解压)

``` shell
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

参考[python_demo](./python_demo) 与 [demo](./demo) 下面的README.md


# 对`modeling_qwen.py`文件代码做调整

1) 第一点修改如下（这是因为TORCH2的算子转ONNX会失败）：

    ``` python
    # SUPPORT_TORCH2 = hasattr(torch, '__version__') and int(torch.__version__.split(".")[0]) >= 2
    SUPPORT_TORCH2 = False
    ```

2) 第二点修改如下（这是因为转ONNX，提示Shape推导失败）：

    ```python
    # attn_weights = attn_weights / torch.full(
    #     [],
    #     size_temp ** 0.5,
    #     dtype=attn_weights.dtype,
    #     device=attn_weights.device,
    # )
    attn_weights = attn_weights / (size_temp ** 0.5)
    ```

3) 第三点修改如下（这段代码全部注释掉，是因为可以直接采用`attention_mask`，避免复杂逻辑，提升性能）：

    ```python
    # if self.use_cache_quantization:
    #     query_length, key_length = query.size(-2), key[0].size(-2)
    # else:
    #     query_length, key_length = query.size(-2), key.size(-2)
    # causal_mask = registered_causal_mask[
    #     :, :, key_length - query_length : key_length, :key_length
    # ]
    # mask_value = torch.finfo(attn_weights.dtype).min
    # mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
    #     attn_weights.device
    # )
    # attn_weights = torch.where(
    #     causal_mask, attn_weights.to(attn_weights.dtype), mask_value
    # )
    ```

4) 第四点修改如下（同上原因）：

    ``` python
    # query_length, key_length = query.size(-2), key.size(-2)
    # causal_mask = registered_causal_mask[
    #     :, :, key_length - query_length : key_length, :key_length
    # ]
    # mask_value = torch.finfo(attn_weights.dtype).min
    # mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(
    #     attn_weights.device
    # )
    # attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    ```

5) 第五点修改，将如下代码移至`if layer_past is not None:`之前：

    ``` python
    if use_cache:
        present = (key, value)
    else:
        present = None
    ```

    这是因为kv cache只用输出1个单位就可以了，不用全部输出。提升效率。