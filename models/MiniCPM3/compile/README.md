# Command

## Export onnx

```shell
pip install -r requirements.txt
cp files/MiniCPM3-4B/modeling_minicpm.py ${your_torch_model}/modeling_minicpm.py
```
your_torch_model是你模型下载的位置，比如 MiniCPM3-4B/

```shell
python3 export_onnx.py --model_path your_torch_model --seq_length 8192 --device cpu
```
* 风险点：尤其注意，如果使用--device cpu在cpu上导出，使用的精度是float32，与训练精度bfloat16不一致，可能导致精度问题
* 如果有cuda，建议使用cuda导出

## Compile bmodel
使用io_alone, int4精度
```shell
./compile.sh --mode int4 --name minicpm3-4b --addr_mode io_alone --seq_length 8192
```
使用io_alone, int8精度
```shell
./compile.sh --mode int8 --name minicpm3-4b --addr_mode io_alone --seq_length 8192
```

### 下载迁移好的模型
也可以直接下载编译好的模型，不用自己编译
```shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm3-4b_int4_seq512_1dev.bmodel
```

## python demo

请见python_demo里面的README

### modeling_minicpm.py代码修改

#### 第一处：MiniCPM3Model

* 在初始化中添加
```python
config._attn_implementation = "eager"
```
目的是不使用torch的flash attention或者sdpa attenion，导出原始的attention结构便于在mlir编译时匹配优化pattern。

#### 第二处：修改旋转位置编码
原代码：
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
修改后
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

#### 第三处：MiniCPMAttention

* 主要修改了past_key_value和position_embedding的输入方式，将position_embedding进行常量折叠，导出onnx结构。 

* 其次修改了attention计算时的一些permute和concat操作，用于后续mlir编译模型时的pattern匹配和简化模型

* 具体修改可对比原始模型文件
