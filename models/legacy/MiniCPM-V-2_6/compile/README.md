# Command

## Export onnx

```shell
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install transformers_stream_generator einops tiktoken accelerate transformers==4.40.0
cp files/MiniCPM-V-2_6/modeling_qwen2.py /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/
cp files/MiniCPM-V-2_6/resampler.py your_torch_model
cp files/MiniCPM-V-2_6/modeling_navit_siglip.py your_torch_model
```
`your_torch_model` is the location of your model.
```shell
python3 export_onnx.py --model_path your_torch_model --seq_length 1024 --device cpu --image_file ../python_demo/test0.jpg
```
* image_file: `image_file` is the path of a real image. When exporting the model, the input size will be fixed to the size of this image. `Please enter your actual image for image_file`
* Variable image sizes are not currently supported

## Compile bmodel
Use io_alone
```
./compile.sh --mode int4 --name minicpmv26 --seq_length 1024
```

### Download the migrated model
You can also directly download the precompiled model instead of compiling it yourself
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpmv26_bm1684x_int4_seq1024_imsize448.bmodel
```

### python demo

See the README inside python_demo

### Modifications to modeling_qwen2.py

#### First change: modify the rotary position embedding
Original code:
```python
cos = cos[position_ids].unsqueeze(unsqueeze_dim)
sin = sin[position_ids].unsqueeze(unsqueeze_dim)
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```
After modification
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

* The other modifications are the same as those for Qwen1_5
