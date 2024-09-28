![image](./assets/sophgo_chip.png)

# Llama3.1

本项目实现BM1684X部署语言大模型[Llama3.1-8B-instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

下文中默认是PCIE环境；如果是SoC环境，按提示操作即可。

# 目录说明
```
.
├── README.md
├── compile
│   ├── compile.sh                          #用来编译TPU模型的脚本
│   ├── export_onnx.py                      #用来导出onnx的脚本
│   └── files                               #用于替换原模型的文件
├── python_demo
│   ├── chat.cpp                            #主程序文件
│   └── pipeline.py                         #Llama3.1 python_demo的执行脚本
├── requirements.txt                        #环境配置所需安装的wheel包
├── run_demo.sh                             #自动测试脚本
└── token_config                            #分词器
    ├── special_tokens_map.json
    ├── tokenizer.json
    └── tokenizer_config.json
```
----------------------------

#  自动化推理脚本



# 【阶段一】模型编译

## 注意点
* 模型编译必须要在docker内完成，无法在docker外操作

### 步骤一：模型下载
虽然Llama3模型允许商业开源，但是模型下载需要想Meta提交使用申请，因此测试模型时可以参考[ModelScope提供的模型权重](https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct/summary)进行下载，或者通过Huggingface申请Meta License进行下载。


### 步骤二：下载docker

下载docker，启动容器，如下：

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

### 步骤三：下载TPU-MLIR代码并编译

``` shell
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```
* PS：重新进入docker环境并且需要编译模型时，必须在此路径下执行上述`source ./envsetup.sh` 和 `./build.sh`才能完成后续模型编译。

### 步骤四：对齐模型环境

``` shell
pip install -r requirements.txt
cp ./compile/files/Meta-Llama-3.1-8B-Instruct/modeling_llama.py /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
```
同时将`./compile/files/Meta-Llama-3.1-8B-Instruct/config.json` 替换下载好的`Llama-3.1-8B-Instruct`路径下的同名文件。

* PS：不一定是/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py这个路径，建议替换前先pip show transformers查看一下

### 步骤五：生成onnx文件

``` shell
cd compile
python export_onnx.py --model_path your_model_path --seq_length 512
```

* PS1：your_model_path 指的是原模型下载后的地址, 如:"../../Meta-Llama-3.1-8B-Instruct"。
* PS2：默认导出sequence length为512的模型

### 步骤六：生成bmodel文件

生成单芯模型

``` shell
./compile.sh --mode int8 --name llama3.1-8b --seq_length 512 # same as int4
```

* PS1：生成bmodel耗时大概3小时以上，建议64G内存以及200G以上硬盘空间，不然很可能OOM或者no space left
* PS2：如果想要编译llama3.1-8b，则--name必须为llama3.1-8b
* PS3：目前给定的lib_pcie和lib_soc部分仅包含单芯的动态库，多芯部分会在后续更新
* PS4：步骤三到步骤六可以通过运行compile文件夹下的run_compile.sh完成，具体命令是：
``` shell
./run_compile.sh --model_name llama3.1-8b --seq_length 512 --model_path your model path --tpu_mlir_path your tpu_mlir path
```
如果没有填写model_path，脚本会从modelscope下载模型，如果没有填写tpu_mlir_path，脚本会通过dfss下载对应的tpu_mlir压缩包并解压
----------------------------

# 阶段二：可执行文件生成

## 编译程序(Python Demo版本)
执行如下编译，(PCIE版本与SoC版本相同)：

```shell
cd python_demo
mkdir build
cd build
cmake ..
make
cp *chat* ..
```

## 模型推理(Python Demo版本)
```shell
cd ./python_demo
python3 pipeline.py -m your_model_path -t ../token_config --devid your_devid
```
其它可用参数可以通过`pipeline.py` 或者执行如下命令进行查看 
```shell
python3 pipeline.py --help
```

## 常见问题

#### 对源码做了哪些修改：

- 在`modeling_llama.py`文件中：

增加了

```python
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
```
利用 attention_mask 计算批次中每个序列的实际长度（未填充部分）

- 在LlamaRotaryEmbedding类中

增加了

```python
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos()
        self.sin = emb.sin()
```

- 将LlamaRotaryEmbedding类如下代码：

```python
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

修改为：

```python
    def forward(self, x, position_ids=None, seq_len=None):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)
        return (
            self.cos[:seq_len, :].to(dtype=x.dtype),
            self.sin[:seq_len, :].to(dtype=x.dtype),
        )
```

目的是预先计算旋转位置编码，这样在模型的前向传播中就可以直接使用这些编码，而不需要每次都重新计算。这种方法可以提高模型的计算效率。

- 在`modeling_llama.py`文件中如下代码：

```python
    class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
        """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

        def __init__(self, *args, **kwargs):
            logger.warning_once(
                "`LlamaLinearScalingRotaryEmbedding` is deprecated an will be removed in v4.45. Please use "
                "`LlamaRotaryEmbedding`, which now also does linear scaling (simply pass the model config to __init__)."
            )
            kwargs["rope_type"] = "linear"
            super().__init__(*args, **kwargs)


    class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
        """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

        def __init__(self, *args, **kwargs):
            logger.warning_once(
                "`LlamaDynamicNTKScalingRotaryEmbedding` is deprecated an will be removed in v4.45. Please use "
                "`LlamaRotaryEmbedding`, which now also does dynamic ntk scaling (simply pass the model config to "
                "__init__)."
            )
            kwargs["rope_type"] = "dynamic"
            super().__init__(*args, **kwargs)
```

修改为：

```python
    class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
        """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

        def forward(self, x, position_ids):
            # difference to the original RoPE: a scaling factor is aplied to the position ids
            position_ids = position_ids.float() / self.scaling_factor
            cos, sin = super().forward(x, position_ids)
            return cos, sin


    class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
        """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

        def forward(self, x, position_ids):
            # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
            seq_len = torch.max(position_ids) + 1
            if seq_len > self.max_position_embeddings:
                base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
                ) ** (self.dim / (self.dim - 2))
                inv_freq = 1.0 / (
                    base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
                )
                self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

            cos, sin = super().forward(x, position_ids)
            return cos, sin
```

- 在`modeling_llama.py`文件中如下代码：

```python
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
```

修改为：

```python
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
```

- 在`modeling_llama.py`文件中如下代码：

```python
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
```

修改为：

```python
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, slen, num_key_value_heads, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states.transpose(1,2)
```

- 在`modeling_llama.py`文件中如下代码：

```python
    class LlamaAttention(nn.Module):
        ......

            if self.config.pretraining_tp > 1:
                key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
                query_slices = self.q_proj.weight.split(
                    (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
                )
                key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
                value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

                query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
                query_states = torch.cat(query_states, dim=-1)

                key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
                key_states = torch.cat(key_states, dim=-1)

                value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
                value_states = torch.cat(value_states, dim=-1)

            else:
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()

            attn_output = attn_output.reshape(bsz, q_len, -1)

            if self.config.pretraining_tp > 1:
                attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
                o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
                attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
            else:
                attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value
```

修改为：

```python
    class LlamaAttention(nn.Module):
        ......
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

            kv_seq_len = key_states.shape[-3]
            if past_key_value is not None:
                cos, sin = self.rotary_emb(value_states, seq_len= kv_seq_len + past_key_value[0].shape[-3] -1)
            else:
                cos, sin = self.rotary_emb(value_states, seq_len = kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            past_kv = (key_states, value_states)
            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=1)
                value_states = torch.cat([past_key_value[1], value_states], dim=1)
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states.transpose(1, 2), key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            
            # A = (Q @ K^T) @ V
            attn_output = torch.matmul(attn_weights, value_states)
            
            # O = FFN(A)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

            attn_weights = None

            return attn_output, attn_weights, past_kv
```

- 在`modeling_llama.py`文件中如下代码：

```python
    class LlamaFlashAttention2(LlamaAttention):
        ......
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        ......
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
```

修改为：

```python
    class LlamaFlashAttention2(LlamaAttention):
        ......
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            past_key_value = getattr(self, "past_key_value", past_key_value)

        ......
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
```

新增了：

```python
        def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
        ):
            if not self._flash_attn_uses_top_left_mask:
                causal = self.is_causal
            else:
                # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
                causal = self.is_causal and query_length != 1

            # Contains at least one padding token in the sequence
            if attention_mask is not None:
                batch_size = query_states.shape[0]
                query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                    query_states, key_states, value_states, attention_mask, query_length
                )

                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )

                attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
            else:
                attn_output = flash_attn_func(
                    query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
                )

            return attn_output
        def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
            indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
            batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

            key_layer = index_first_axis(
                key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
            )
            value_layer = index_first_axis(
                value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
            )
            if query_length == kv_seq_len:
                query_layer = index_first_axis(
                    query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
                )
                cu_seqlens_q = cu_seqlens_k
                max_seqlen_in_batch_q = max_seqlen_in_batch_k
                indices_q = indices_k
            elif query_length == 1:
                max_seqlen_in_batch_q = 1
                cu_seqlens_q = torch.arange(
                    batch_size + 1, dtype=torch.int32, device=query_layer.device
                )  # There is a memcpy here, that is very bad.
                indices_q = cu_seqlens_q[:-1]
                query_layer = query_layer.squeeze(1)
            else:
                # The -q_len: slice assumes left padding.
                attention_mask = attention_mask[:, -query_length:]
                query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

            return (
                query_layer,
                key_layer,
                value_layer,
                indices_q,
                (cu_seqlens_q, cu_seqlens_k),
                (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
            )
```

- 在`modeling_llama.py`文件中如下代码：

```python
    class LlamaFlashAttention2(LlamaAttention):
        ......
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        ......
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
```

修改为：

```python
    class LlamaFlashAttention2(LlamaAttention):
        ......
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            past_key_value = getattr(self, "past_key_value", past_key_value)

        ......
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
```

新增了：

```python
        def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
        ):
            if not self._flash_attn_uses_top_left_mask:
                causal = self.is_causal
            else:
                # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
                causal = self.is_causal and query_length != 1

            # Contains at least one padding token in the sequence
            if attention_mask is not None:
                batch_size = query_states.shape[0]
                query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                    query_states, key_states, value_states, attention_mask, query_length
                )

                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )

                attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
            else:
                attn_output = flash_attn_func(
                    query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
                )

            return attn_output
        def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
            indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
            batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

            key_layer = index_first_axis(
                key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
            )
            value_layer = index_first_axis(
                value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
            )
            if query_length == kv_seq_len:
                query_layer = index_first_axis(
                    query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
                )
                cu_seqlens_q = cu_seqlens_k
                max_seqlen_in_batch_q = max_seqlen_in_batch_k
                indices_q = indices_k
            elif query_length == 1:
                max_seqlen_in_batch_q = 1
                cu_seqlens_q = torch.arange(
                    batch_size + 1, dtype=torch.int32, device=query_layer.device
                )  # There is a memcpy here, that is very bad.
                indices_q = cu_seqlens_q[:-1]
                query_layer = query_layer.squeeze(1)
            else:
                # The -q_len: slice assumes left padding.
                attention_mask = attention_mask[:, -query_length:]
                query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

            return (
                query_layer,
                key_layer,
                value_layer,
                indices_q,
                (cu_seqlens_q, cu_seqlens_k),
                (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
            )
```

- 在`modeling_llama.py`文件中如下代码：
```python
    class LlamaPreTrainedModel(PreTrainedModel):
```
新增了：
```python
        def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None):
            if self.config._attn_implementation == "flash_attention_2" and cache_cls == StaticCache:
                raise ValueError(
                    "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                    "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
                )

            for layer in self.model.layers:
                device = layer.input_layernorm.weight.device
                if hasattr(self.config, "_pre_quantization_dtype"):
                    dtype = self.config._pre_quantization_dtype
                else:
                    dtype = layer.self_attn.o_proj.weight.dtype
                layer.self_attn.past_key_value = cache_cls(
                    self.config, max_batch_size, max_cache_len, device=device, dtype=dtype
                )

        def _reset_cache(self):
            for layer in self.model.layers:
                layer.self_attn.past_key_value = None
```

- 在`modeling_llama.py`文件中如下代码：
```python
    class LlamaModel(LlamaPreTrainedModel):
        def forward(
        ...
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        ...)
        ...
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        ...
        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()
```
修改为：
```python
    class LlamaModel(LlamaPreTrainedModel):
        def forward(
        ...
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        ...)
        ...
        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)
        ...
        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
```

- 在`modeling_llama.py`文件中如下代码：
```python
    def _update_causal_mask(
        ...
        past_key_values: Cache,
        output_attentions: bool,
    ):

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
```
修改为

```python
    def _update_causal_mask(
        ...
        past_seen_tokens: int,
    ):

        if self.config._attn_implementation == "sdpa":
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask, inputs_embeds=input_tensor, past_key_values_length=past_seen_tokens
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if hasattr(getattr(self.layers[0], "self_attn", {}), "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice
```

- 在`modeling_llama.py`文件中如下代码：
```python
    class LlamaForCausalLM(LlamaPreTrainedModel):
        ...
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        ...
                if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
```
修改为
```python
    class LlamaForCausalLM(LlamaPreTrainedModel):
        ...
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        ...
                has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None
        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```
- 在`modeling_llama.py`文件中如下代码：
```python
    @add_start_docstrings(
        """
        The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
        output) e.g. for Named-Entity-Recognition (NER) tasks.
        """,
        LLAMA_START_DOCSTRING,
    )
    class LlamaForTokenClassification(LlamaPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.model = LlamaModel(config)
            if getattr(config, "classifier_dropout", None) is not None:
                classifier_dropout = config.classifier_dropout
            elif getattr(config, "hidden_dropout", None) is not None:
                classifier_dropout = config.hidden_dropout
            else:
                classifier_dropout = 0.1
            self.dropout = nn.Dropout(classifier_dropout)
            self.score = nn.Linear(config.hidden_size, config.num_labels)

            # Initialize weights and apply final processing
            self.post_init()

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def set_input_embeddings(self, value):
            self.model.embed_tokens = value

        @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, TokenClassifierOutput]:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            """
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            sequence_output = self.dropout(sequence_output)
            logits = self.score(sequence_output)

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
```
进行删除