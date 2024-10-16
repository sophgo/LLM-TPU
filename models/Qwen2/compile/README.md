# Command

## Export onnx

```shell
pip install transformers_stream_generator einops tiktoken accelerate torch==2.0.1+cpu torchvision==0.15.2 transformers==4.41.2
cp files/Qwen2-7B-Instruct/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/
```
your_torch_model是你模型的位置
```shell
python3 export_onnx.py --model_path your_torch_model --seq_length 8192 --device cpu
```
* 风险点：尤其注意，如果使用--device cpu在cpu上导出，使用的精度是float32，与训练精度bfloat16不一致，可能导致精度问题
* 如果有cuda，建议使用cuda导出

## Compile bmodel
使用io_alone
```
./compile.sh --mode int4 --name qwen2-7b --addr_mode io_alone --seq_length 8192
```

编译Qwen2-1.5B
```
./compile.sh --mode int4 --name qwen2-1.5b --addr_mode io_alone --seq_length 8192
```

### 下载迁移好的模型
也可以直接下载编译好的模型，不用自己编译
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2-1.5b_int4_seq8192_1dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2-7b_int4_seq8192_1dev.bmodel
```

### python demo

请见python_demo里面的README

### modeling_qwen2.py代码修改

#### 第一处：修改旋转位置编码
原代码：
```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```
修改后
```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=2):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
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

* 其他修改位置与Qwen1_5相同
