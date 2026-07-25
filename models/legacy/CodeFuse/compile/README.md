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

You need to download tpu-mlir in advance here.
```shell
pushd /path_to/tpu-mlir
source envsetup.sh
popd
```

### compile basic bmodel
```shell
./compile.sh --mode int4 --name codefuse-7b --addr_mode io_alone --seq_length 2048
```


## Adjust the Code in the `modeling_qwen.py` File

1) The first modification is as follows (to enable constant folding and prevent rotary_emb from being updated):

    ```python
    rotary_pos_emb = self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha).to(
        hidden_states.device
    )
    ```
    Change to
    ```python
    # rotary_pos_emb = self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha).to(
    #     hidden_states.device
    # )
    ```

2) The second modification is as follows (to enable constant folding):
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
    Change to
    ```python
    if rotary_pos_emb is not None:
        # breakpoint()
        # q_pos_emb, k_pos_emb = rotary_pos_emb
        # Slice the pos emb for current inference
        # cur_len = query.shape[1]
        # q_pos_emb = q_pos_emb[:, -cur_len:, :, :]
        # k_pos_emb = k_pos_emb[:, -cur_len:, :, :]
        # query = apply_rotary_pos_emb(query, q_pos_emb)
        # key = apply_rotary_pos_emb(key, k_pos_emb)
        query = apply_rotary_pos_emb(query, rotary_pos_emb)
        key = apply_rotary_pos_emb(key, rotary_pos_emb)
    ```

3) The third modification is as follows (to enable constant folding; the cos and sin built into tpu-mlir will cause accuracy loss):
    ```python
    # def apply_rotary_pos_emb(t, freqs):
    #     if apply_rotary_emb_func is not None:
    #         t_ = t.float()
    #         freqs = freqs.squeeze(0).squeeze(1)
    #         cos = freqs[:, : freqs.shape[-1] // 2].cos()
    #         sin = freqs[:, : freqs.shape[-1] // 2].sin()
    #         output = apply_rotary_emb_func(t_, cos, sin).type_as(t)
    #         return output
    #     else:
    #         rot_dim = freqs.shape[-1]
    #         t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
    #         t_ = t_.float()
    #         t_pass_ = t_pass_.float()
    #         t_ = (t_ * freqs.cos()) + (_rotate_half(t_) * freqs.sin())
    #         return torch.cat((t_, t_pass_), dim=-1).type_as(t)
    ```
    Change to
    ```python
    def apply_rotary_pos_emb(t, freqs):
        cos, sin = freqs
        if apply_rotary_emb_func is not None and t.is_cuda:
            t_ = t.float()
            cos = cos.squeeze(0).squeeze(1)[:, : cos.shape[-1] // 2]
            sin = sin.squeeze(0).squeeze(1)[:, : sin.shape[-1] // 2]
            output = apply_rotary_emb_func(t_, cos, sin).type_as(t)
            return output
        else:
            rot_dim = freqs[0].shape[-1]
            cos, sin = freqs
            t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
            t_ = t_.float()
            t_pass_ = t_pass_.float()
            t_ = (t_ * cos) + (_rotate_half(t_) * sin)
            return torch.cat((t_, t_pass_), dim=-1).type_as(t)
    ```


4) The fourth modification is as follows (to speed up inference and avoid kvcache reuse):
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
    Change to
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

4) The fifth modification is as follows (to avoid softmax overflow):
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
    Change to
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

