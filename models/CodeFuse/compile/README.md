# Command

## Export onnx

```shell
pip install transformers_stream_generator einops tiktoken accelerate transformers==4.32.0
```

### export basic onnx
```shell
python export_onnx.py --model CodeFuse-DevOps-Model-7B-Chat/ --seq_length 2048
```

## Compile bmodel

这里需要提前下载好tpu-mlir
```shell
pushd /path_to/tpu-mlir
source envsetup.sh
popd
```

### compile basic bmodel
```shell
./compile.sh --mode int4 --name codefuse-7b --addr_mode io_alone --seq_length 2048
```


修改点
```python
rotary_pos_emb = self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha).to(
    hidden_states.device
)
```
```python
# rotary_pos_emb = self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha).to(
#     hidden_states.device
# )
```


```python
if rotary_pos_emb is not None:
    q_pos_emb, k_pos_emb = rotary_pos_emb
    # Slice the pos emb for current inference
    cur_len = query.shape[1]
    q_pos_emb = q_pos_emb[:, cur_len, :, :]
    k_pos_emb = k_pos_emb[:, cur_len, :, :]
    query = apply_rotary_pos_emb(query, q_pos_emb)
    key = apply_rotary_pos_emb(key, k_pos_emb)
```
```python
if rotary_pos_emb is not None:
    q_pos_emb, k_pos_emb = rotary_pos_emb
    # Slice the pos emb for current inference
    # cur_len = query.shape[1]
    q_pos_emb = q_pos_emb[:, cur_len, :, :]
    k_pos_emb = k_pos_emb[:, cur_len, :, :]
    query = apply_rotary_pos_emb(query, q_pos_emb)
    key = apply_rotary_pos_emb(key, k_pos_emb)
```


```python
if layer_past is not None:
    past_key, past_value = layer_past[0], layer_past[1]
    key = torch.cat((past_key, key), dim=1)
    value = torch.cat((past_value, value), dim=1)

if use_cache:
    present = (key, value)
else:
    present = None
```
```python
if use_cache:
    present = (key, value)
else:
    present = None

if layer_past is not None:
    past_key, past_value = layer_past[0], layer_past[1]
    key = torch.cat((past_key, key), dim=1)
    value = torch.cat((past_value, value), dim=1)
```


```python
query_length, key_length = query.size(-2), key.size(-2)
causal_mask = self.bias[
    :, :, key_length - query_length : key_length, :key_length
]
mask_value = torch.finfo(attn_weights.dtype).min
mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
    attn_weights.device
)
attn_weights = torch.where(
    causal_mask, attn_weights.to(attn_weights.dtype), mask_value
)

if attention_mask is not None:
    # Apply the attention mask
    attn_weights = attn_weights + attention_mask
```
```python
# query_length, key_length = query.size(-2), key.size(-2)
# causal_mask = self.bias[
#     :, :, key_length - query_length : key_length, :key_length
# ]
# mask_value = torch.finfo(attn_weights.dtype).min
# mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
#     attn_weights.device
# )
# attn_weights = torch.where(
#     causal_mask, attn_weights.to(attn_weights.dtype), mask_value
# )

if attention_mask is not None:
    # Apply the attention mask
    attn_weights = attn_weights + attention_mask
```
