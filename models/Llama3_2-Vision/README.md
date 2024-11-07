# Llama3.2

本项目实现BM1684X部署语言大模型[Llama-3.2-11B-Vision-Instruct](https://www.modelscope.cn/models/LLM-Research/Llama-3.2-11B-Vision-Instruct)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

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
│   └── pipeline.py                         #Llama3.2 vision python_demo的执行脚本
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
虽然Llama3模型允许商业开源，但是模型下载需要想Meta提交使用申请，因此测试模型时可以参考[ModelScope提供的模型权重](https://www.modelscope.cn/models/LLM-Research/Llama-3.2-11B-Vision-Instruct)进行下载，或者通过Huggingface申请Meta License进行下载。


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
cp ./compile/files/Llama-3.2-11B-Vision-Instruct/modeling_mllama.py /usr/local/lib/python3.10/dist-packages/transformers/models/mllama/modeling_mllama.py
```
* PS：不一定是/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_mllama.py这个路径，建议替换前先pip show transformers查看一下

### 步骤五：生成onnx文件

``` shell
cd compile
python export_onnx.py -m your_model_path -s 512
```

* PS1：your_model_path 指的是原模型下载后的地址, 如:"../Llama-3.2-11B-Vision-Instruct"。
* PS2：默认导出sequence length为512的模型

### 步骤六：生成bmodel文件

生成单芯模型

``` shell
./compile.sh --mode int4 --name llama3.2-11b --seq_length 512 # same as int8
```
* PS1：生成bmodel耗时大概3小时以上，建议64G内存以及200G以上硬盘空间，不然很可能OOM或者no space left
* PS2：如果想要编译llama3.2-11b，则--name必须为llama3.2-11b
* PS3：目前给定的lib_pcie和lib_soc部分仅包含单芯的动态库，多芯部分会在后续更新
* PS4：步骤三到步骤六可以通过运行compile文件夹下的run_compile.sh完成，具体命令是：
``` shell
./run_compile.sh --model_name llama3.2-11b --seq_length 512 --model_path your model path --tpu_mlir_path your tpu_mlir path
```
如果没有填写model_path，脚本会从modelscope下载模型，如果没有填写tpu_mlir_path，脚本会通过dfss下载对应的tpu_mlir压缩包并解压
----------------------------

# 【阶段二】可执行文件生成

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
python3 pipeline.py -m your_model_path -i your_image_path -t ../token_config --devid your_devid
```
其它可用参数可以通过`pipeline.py` 或者执行如下命令进行查看 
```shell
python3 pipeline.py --help
```

## 常见问题

#### 对源码做了哪些修改：

- 在`modeling_mllama.py`文件中：

- MllamaVisionModel类中
主要对.shape等操作转换为静态shape，避免模型过大导致常量折叠失败，进而导致动态子网
还有一些scatterND操作，通过改为concat实现，理由同上

- 在`modeling_mllama.py`文件中如下代码：

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

- 在`modeling_mllama.py`文件中如下代码：

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
        hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
        return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim).transpose(1, 2)
```

- 在`modeling_mllama.py`文件中如下代码：

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


目的是减少attention中的permute操作，以便匹配mlir中的一些优化pattern