# README

## Command

### Export onnx

```
pip install transformers==4.46.2 torch==2.4.1+cu121 torchvision==0.19.0+cu121
cp files/modeling_moss.py moss-moon-003-sft/
```

your_torch_model是你模型的路径

```
python3 export_onnx.py --model_path your_torch_model --seq_length 512
```

### Compile bmodel

```
./gen_bmodel.sh --target bm1684x --mode int4 --name moss --addr_mode io_alone
```

### modeling_moss代码修改

#### 第一处

原代码：

```python
mask_value = torch.finfo(attn_weights.dtype).min
mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
attn_weights = torch.where(causal_mask, attn_weights, mask_value)
```

修改后：将这三行代码注释，消去where算子

#### 第二处

原代码：

```python
if layer_past is not None:
    past_key = layer_past[0]
    past_value = layer_past[1]
    key = torch.cat((past_key, key), dim=-2)
    value = torch.cat((past_value, value), dim=-2)
```

修改后：

```python
if layer_past is not None:
    past_key = layer_past[0]
    past_value = layer_past[1]
    past_key=past_key.permute(0, 2, 1, 3)
    past_value=past_value.permute(0, 2, 1, 3)
    key = torch.cat((past_key, key), dim=-2)
    value = torch.cat((past_value, value), dim=-2)
```

对齐past_key和past_value的形状