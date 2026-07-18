# Command

## Export onnx

```shell
pip install transformers_stream_generator einops tiktoken accelerate torch==2.0.1+cpu torchvision==0.15.2 transformers==4.41.2
cp files/Qwen2-7B-Instruct/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/
```
your_torch_model is the location of your model
```shell
python3 export_onnx.py --model_path your_torch_model --seq_length 8192 --device cpu
```

## Compile bmodel
Use io_alone
```
./compile.sh --mode int4 --name qwen2-7b --addr_mode io_alone --seq_length 8192
```

Compile Qwen2-1.5B
```
./compile.sh --mode int4 --name qwen2-1.5b --addr_mode io_alone --seq_length 8192
```

Migrate Qwen2-7B, single chip
``` shell
./run_compile.sh --model_name qwen2-7b --seq_length 4096
```
* If model_path is not specified, the script will download the model from ModelScope
* If tpu_mlir_path is not specified, the script will download the corresponding tpu_mlir archive via dfss and extract it
* If num_device is not specified, it defaults to a single chip
* If mode is not specified, it defaults to int4, W4BF16 quantization

### Download the migrated model
You can also directly download the compiled model instead of compiling it yourself
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2-1.5b_int4_seq8192_1dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2-7b_int4_seq8192_1dev.bmodel
```

### python demo

See the README in python_demo

### modeling_qwen2.py code modifications

#### First change: modify the rotary position embedding
Original code:
```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```
After modification
```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=2):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

#### Second change: modify repeat_kv

Original code:
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

After modification
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

* Other modification locations are the same as Qwen1_5
