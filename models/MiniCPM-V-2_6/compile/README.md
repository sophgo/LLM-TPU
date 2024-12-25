# Command

## Export onnx

```shell
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install transformers_stream_generator einops tiktoken accelerate transformers==4.40.0
cp files/MiniCPM-V-2_6/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/
cp files/MiniCPM-V-2_6/resampler.py your_torch_model
cp files/MiniCPM-V-2_6/modeling_navit_siglip.py your_torch_model
```
your_torch_model是你模型的位置
```shell
python3 export_onnx.py --model_path your_torch_model --seq_length 1024 --device cpu --image_file ../python_demo/test0.jpg
```
* image_file：image_file为真实图片的路径，导出模型时，输入size会固定为该图片的size。`image_file请输入你实际的图片`
* 目前不支持图片size可变

## Compile bmodel
使用io_alone
```
./compile.sh --mode int4 --name minicpmv26 --seq_length 1024
```

### 下载迁移好的模型
也可以直接下载编译好的模型，不用自己编译
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpmv26_bm1684x_int4_seq1024_imsize448.bmodel
```

### python demo

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

* 其他修改位置与Qwen1_5相同
