# Command

## Export onnx

```shell
pip install transformers_stream_generator einops tiktoken accelerate torch==2.0.1+cpu torchvision==0.15.2 transformers==4.45.2
cp files/Qwen2.5-7B-Instruct/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/
```
your_torch_model是你模型的位置
```shell
python3 export_onnx.py --model_path your_torch_model --seq_length 8192 --device cpu
```
* 风险点：尤其注意，如果使用--device cpu在cpu上导出，使用的精度是float32，与训练精度bfloat16不一致，可能导致精度问题
* 如果有cuda，建议使用cuda导出

## Compile bmodel
使用io_alone
```shell
./compile.sh --mode int4 --name qwen2.5-7b --addr_mode io_alone --seq_length 8192
```

使用dynamic
```shell
./compile.sh --mode int4 --name qwen2.5-7b --addr_mode io_alone --dynamic 1 --seq_length 8192
```

### 下载迁移好的模型
也可以直接下载编译好的模型，不用自己编译
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-7b_int4_seq8192_1dev.bmodel
```

### 全流程编译脚本
以上步骤可以通过运行compile文件夹下的run_compile.sh完成，具体命令是：
``` shell
./run_compile.sh --model_name qwen2.5-7b --seq_length 512 --model_path your model path --tpu_mlir_path your tpu_mlir path
```
如果没有填写model_path，脚本会从modelscope下载模型；
如果没有填写tpu_mlir_path，脚本会通过dfss下载对应的tpu_mlir压缩包并解压


## python demo

请见python_demo里面的README

### modeling_qwen2.py代码修改

#### 第一处：修改旋转位置编码
原代码：
```python
cos = cos[position_ids].unsqueeze(unsqueeze_dim)
sin = sin[position_ids].unsqueeze(unsqueeze_dim)
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```
修改后
```python
# The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
cos = cos.transpose(1, 2)
sin = sin.transpose(1, 2)
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```

#### 第二处：修改repeat_kv

原代码：
```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

修改后
```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)
```

* 其他修改位置与Qwen2相同