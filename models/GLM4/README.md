![image](./assets/sophgo_chip.png)

# ChatGLM4

本项目实现BM1684X部署语言大模型[glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

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
│   ├── pipeline.py                         #ChatGLM4 python demo的执行脚本
│   └── web_demo.py                         #ChatGLM4 web demo的执行脚本
├── requirements.txt                        #环境配置所需安装的wheel包
├── run_demo.sh                             #自动测试脚本
└── token_config                            #分词器
    ├── tokenization_chatglm.py
    ├── tokenizer_config.json
    └── tokenizer.model
```
----------------------------

#  自动化推理脚本



# 【阶段一】模型编译

如果不打算自己编译模型，可以直接下载编译好的模型：
```bash
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/glm4-9b_int4_1dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/glm4-9b_int8_1dev.bmodel
```
## 注意点
* 模型编译必须要在docker内完成，无法在docker外操作。

### 步骤一：下载docker

下载docker，启动容器，如下：

```bash
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
* PS：本repo `LLM-TPU`需在当前目录内

### 步骤二：下载TPU-MLIR代码并安装

``` shell
pip3 install dfss  --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/tpu-mlir.tar.gz
tar zxvf tpu-mlir.tar.gz
cd tpu-mlir_v1.8.beta.0-134-g859a6f517-20240801
source ./envsetup.sh
cd ..
```
* PS：重新进入docker环境并且需要编译模型时，必须在此路径下执行上述`source ./envsetup.sh`才能完成后续模型编译。

### 步骤三：模型下载
ChatGLM4模型允许商业开源，可以通过Huggingface官网下载[glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat)。
如果无法从官网下载，这里也提供一个下载好的压缩包。
```bash
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/glm-4-9b-chat-torch.zip
unzip glm-4-9b-chat-torch.zip
```

下载完`glm-4-9b-chat`官方库后，您还需要设置`ChatGLM4_PATH`环境变量，模型导出时会使用到。
```bash
export ChatGLM4_PATH=$PWD/glm-4-9b-chat
```

### 步骤四：对齐模型环境

```bash
sudo apt-get update
sudo apt-get install pybind11-dev
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cp ./compile/files/glm-4-9b-chat/modeling_chatglm.py $ChatGLM4_PATH
cp ./compile/files/glm-4-9b-chat/config.json $ChatGLM4_PATH
```

### 步骤五：生成onnx文件

```bash
cd compile
python export_onnx.py --model_path $ChatGLM4_PATH --seq_length 512
```
* PS：默认导出sequence length为512的模型。导出其它长度的模型，还需同步修改`$ChatGLM4_PATH/config.json`中的`seq_length`参数。

### 步骤六：生成bmodel文件

生成单芯模型

```bash
./compile.sh --mode int4 --name glm4-9b --seq_length 512 --addr_mode io_alone
```
生成W8A16量化的模型
```bash
./compile.sh --mode int8 --name glm4-9b --seq_length 512 --addr_mode io_alone
```
生成8192长度的模型
```bash
./compile.sh --mode int8 --name glm4-9b --seq_length 8192 --addr_mode io_alone
```


<!-- 生成双芯模型

```bash
./compile.sh --mode int4 --num_device 2 --name glm4-9b --seq_length 512 # same as int8
``` -->

* PS1：生成bmodel耗时大概3小时以上，建议64G内存以及200G以上硬盘空间，不然很可能OOM或者no space left；
* PS2：如果想要编译glm4-9b，则--name必须为glm4-9b。
<!-- * PS3：目前给定的lib_pcie和lib_soc部分仅包含单芯的动态库，多芯部分会在后续更新。 -->

----------------------------

# 阶段二：可执行文件生成

## 编译程序(Python Demo版本)
执行如下编译，(PCIE版本与SoC版本相同)：

```bash
cd python_demo
mkdir build && cd build
cmake ..
make
cp *chat* ..
```

## 模型推理(Python Demo版本)
```bash
cd ./python_demo
python3 pipeline.py --model_path glm4-9b_int4_1dev.bmodel --tokenizer_path ../token_config --devid your_devid
```
其它可用参数可以通过`pipeline.py`或者执行如下命令进行查看。
```bash
python3 pipeline.py --help
```

## web demo
```bash
python3 web_demo.py --model_path glm4-9b_int4_1dev.bmodel --tokenizer_path ../token_config --devid 0
```

### modeling_chatglm.py代码修改

#### 第一处：修改旋转位置编码
原代码：
```python
@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)
```

修改后代码：
```python
# @torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, sq, nq, hn]
    b, sq, nq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    xshaped = x.reshape(b, sq, nq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, sq, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)
```

* `@torch.jit.script`注释掉是为了方便打断点
* 修改为`b, sq, nq, hn = x.size(0), x.size(1), x.size(2), x.size(3)`是因为在前面将维度变为[batch_size, seq_length, head_num, head_dim]，这里需要适应前面的修改
* 其他修改原因等同上一点

#### 第二处：pytorch_major_version

原代码：
```python
if pytorch_major_version >= 2:
```

修改后
```python
if False:
```

* 避免走flash_attention，flash_attention之后在tpu-mlir中会进行图匹配，最后匹配为FAttentionOP，但是这里不能走FAttention

#### 第三处：softmax部分

原代码：
```python
# [b, np, sq, sk]
output_size = (query_layer.size(0), query_layer.size(1), query_layer.size(2), key_layer.size(2))

# [b, np, sq, hn] -> [b * np, sq, hn]
query_layer = query_layer.view(output_size[0] * output_size[1], output_size[2], -1)
# [b, np, sk, hn] -> [b * np, sk, hn]
key_layer = key_layer.view(output_size[0] * output_size[1], output_size[3], -1)

# preallocting input tensor: [b * np, sq, sk]
matmul_input_buffer = torch.empty(
    output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype,
    device=query_layer.device
)

# Raw attention scores. [b * np, sq, sk]
matmul_result = torch.baddbmm(
    matmul_input_buffer,
    query_layer,  # [b * np, sq, hn]
    key_layer.transpose(1, 2),  # [b * np, hn, sk]
    beta=0.0,
    alpha=(1.0 / self.norm_factor),
)

# change view to [b, np, sq, sk]
attention_scores = matmul_result.view(*output_size)
```

修改后
```python
matmul_result = torch.matmul(query_layer.transpose(1,2), key_layer.transpose(1, 2).transpose(2,3))
attention_scores = matmul_result * (1.0 / self.norm_factor)

```

* 这里的修改非常重要
* 去掉`matmul_input_buffer = torch.empty`是因为这会导致转onnx的时候报`>2G`的bug
* 其他修改点是因为输入变成了[batch_size, seq_length, head_num, head_dim]，所以要修改为匹配这种shape的计算

#### 第四处：attention_mask取极值部分

原代码：
```python
attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
```

修改后
```python
直接注释掉
```

* 用求和来代替masked_fill，这是因为芯片后端对masked_fill支持不太好



#### 第五处：（QK）*K

原代码：
```python
output_size = (value_layer.size(0), value_layer.size(1), query_layer.size(1), value_layer.size(3))
# change view [b * np, sk, hn]
value_layer = value_layer.view(output_size[0] * output_size[1], value_layer.size(2), -1)
# change view [b * np, sq, sk]
attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
# matmul: [b * np, sq, hn]
context_layer = torch.bmm(attention_probs, value_layer)
# change view [b, np, sq, hn]
context_layer = context_layer.view(*output_size)
```

修改后
```python
context_layer = torch.matmul(attention_probs, value_layer.transpose(1, 2))
```

* 因为输入变成了[batch_size, seq_length, head_num, head_dim]，所以要修改为匹配这种shape的计算

#### 第六处：QKV输入的处理，输出的处理

原代码位于369~411行
```python
# [b, sq, np, hn] -> [b, np, sq, hn]
query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

# apply relative positional encoding (rotary embedding)
if rotary_pos_emb is not None:
    query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
    key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

if use_cache:
    present_kv_cache = (key_layer, value_layer)
else:
    present_kv_cache = None

# adjust key and value for inference
if kv_cache is not None:
    cache_k, cache_v = kv_cache
    key_layer = torch.cat((cache_k, key_layer), dim=2)
    value_layer = torch.cat((cache_v, value_layer), dim=2)
# if use_cache:
#     kv_cache = (key_layer, value_layer)
#     # if kv_cache is None:
#     #     kv_cache = torch.cat((key_layer.unsqueeze(0).unsqueeze(0), value_layer.unsqueeze(0).unsqueeze(0)), dim=1)
#     # else:
#     #     kv_cache = (key_layer, value_layer)
# else:
#     kv_cache = None


if self.multi_query_attention:
    key_layer = key_layer.unsqueeze(2)
    key_layer = key_layer.expand(
        -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
    )
    key_layer = key_layer.contiguous().view(
        key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
    )
    value_layer = value_layer.unsqueeze(2)
    value_layer = value_layer.expand(
        -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
    )
    value_layer = value_layer.contiguous().view(
        value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
    )
```

修改后位于394~444行
```python
# [b, sq, np, hn] -> [b, np, sq, hn]
# query_layer, key_layer, value_layer = [k.transposes(1, 2) for k in [query_layer, key_layer, value_layer]]

# apply relative positional encoding (rotary embedding)
if rotary_pos_emb is not None:
    query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
    key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

if use_cache:
    present_kv_cache = (key_layer, value_layer)
else:
    present_kv_cache = None

# adjust key and value for inference
if kv_cache is not None:
    cache_k, cache_v = kv_cache
    # key_layer = torch.cat((cache_k, key_layer), dim=2)
    # value_layer = torch.cat((cache_v, value_layer), dim=2)
    key_layer = torch.cat((cache_k, key_layer), dim=1)
    value_layer = torch.cat((cache_v, value_layer), dim=1)

# if self.multi_query_attention:
#     key_layer = key_layer.unsqueeze(2)
#     key_layer = key_layer.expand(
#         -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
#     )
#     key_layer = key_layer.contiguous().view(
#         key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
#     )
#     value_layer = value_layer.unsqueeze(2)
#     value_layer = value_layer.expand(
#         -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
#     )
#     value_layer = value_layer.contiguous().view(
#         value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
#     )

if self.multi_query_attention:
    key_layer = key_layer.unsqueeze(3)
    key_layer = key_layer.expand(
        -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
    )
    key_layer = key_layer.contiguous().view(
        key_layer.size()[:2] + (self.num_attention_heads_per_partition,) + key_layer.size()[4:]
    )
    value_layer = value_layer.unsqueeze(3)
    value_layer = value_layer.expand(
        -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
    )
    value_layer = value_layer.contiguous().view(
        value_layer.size()[:2] + (self.num_attention_heads_per_partition,) + value_layer.size()[4:]
    )

```

* 因为输入变成了[batch_size, seq_length, head_num, head_dim]，所以要修改为匹配这种shape的计算
* 之前的结构是QKV proj前后会有一个permute算子，一共三个permute，但是这样与Qwen系列的结构不同，Qwen系列是FAttention前后会有三个permute。由于这种不同，导致lowering的时候无法匹配为FAttentionOp，因此需要将QKV proj的permute下沉，使其结构和Qwen系列结构保持一致
* 在decode阶段，输出shape从[batch_size, seq_length, head_num, head_dim]改为[batch_size, 1, head_num, head_dim]，避免ConcatOp和搬运