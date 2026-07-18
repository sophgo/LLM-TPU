# Command

## Export onnx

```shell
pip install -r requirements.txt
cp files/MiniCPM3-4B/modeling_minicpm.py ${your_torch_model}/modeling_minicpm.py
```
your_torch_model is where your model is downloaded, e.g. MiniCPM3-4B/

```shell
python3 export_onnx.py --model_path your_torch_model --seq_length 8192 --device cpu
```

## Compile bmodel
Use io_alone, int4 precision
```shell
./compile.sh --mode int4 --name minicpm3-4b --addr_mode io_alone --seq_length 8192
```
Use io_alone, int8 precision
```shell
./compile.sh --mode int8 --name minicpm3-4b --addr_mode io_alone --seq_length 8192
```

### Download the Ported Model
You can also directly download the compiled model instead of compiling it yourself
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm3-4b_int4_seq512_1dev.bmodel
```

## python demo

See the README in python_demo

### modeling_minicpm.py Code Modifications

#### First: MiniCPM3Model

* Add the following during initialization
```python
config._attn_implementation = "eager"
```
The purpose is to avoid using torch's flash attention or sdpa attention, and to export the original attention structure so that optimization patterns can be matched during mlir compilation.

#### Second: Modify the Rotary Position Embedding
Original code:
```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    orig_dtype = k.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_fp32 = q.to(dtype=torch.float32, device=q.device)
    k_fp32 = k.to(dtype=torch.float32, device=k.device)
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)
```
After modification
```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2):
    orig_dtype = k.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, seq_len, 1, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, seq_len, 1, dim]
    q_fp32 = q.to(dtype=torch.float32, device=q.device)
    k_fp32 = k.to(dtype=torch.float32, device=k.device)
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)
```

#### Third: MiniCPMAttention

* Mainly modified the input method of past_key_value and position_embedding, performing constant folding on position_embedding to export the onnx structure.

* Also modified some permute and concat operations in the attention computation, for pattern matching during subsequent mlir compilation and to simplify the model

* For details, compare with the original model file
