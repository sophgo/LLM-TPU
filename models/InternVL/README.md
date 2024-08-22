# InternVL-Chat

本工程实现BM1684X部署多模态大模型[InternVL2-4B](https://huggingface.co/OpenGVLab/InternVL2-4B)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

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

# 对`modeling_phi3.py`文件代码做调整

（也可以直接用compile/files/Phi-3-mini-128k-instruct文件下的modeling_phi3.py替换原模型中的modeling_phi3.py）

1) 第一点修改如下（这是因为transpose计算量较大）：

    ``` python
        #query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        #key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        #value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

    ```

2) 第二点修改如下：

    ```python
        #kv_seq_len = key_states.shape[-2]
        #if past_key_value is not None:
        #    if self.layer_idx is None:
        #    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        #cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        kv_seq_len = key_states.shape[-3]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-3]

        if past_key_value is not None:
          cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len-1)
        else:
          cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
    ```

3) 第三点修改如下：

    ```python
        #if past_key_value is not None:
        #    cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        #    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_kv = (key_states, value_states)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)

    ```