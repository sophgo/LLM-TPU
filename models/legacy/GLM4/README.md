![image](./assets/sophgo_chip.png)

# ChatGLM4

This project deploys the large language model [glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat) on BM1684X. The model is converted to bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed to the BM1684X in a PCIE environment or an SoC environment using C++ code.

The following text assumes a PCIE environment by default; if you are in an SoC environment, just follow the prompts.

# Directory Structure
```
.
├── README.md
├── compile
│   ├── compile.sh                          # script used to compile the TPU model
│   ├── export_onnx.py                      # script used to export onnx
│   └── files                               # files used to replace those in the original model
├── python_demo
│   ├── chat.cpp                            # main program file
│   ├── pipeline.py                         # execution script for the ChatGLM4 python demo
│   └── web_demo.py                         # execution script for the ChatGLM4 web demo
├── requirements.txt                        # wheel packages required for environment setup
├── run_demo.sh                             # automated test script
└── token_config                            # tokenizer
    ├── tokenization_chatglm.py
    ├── tokenizer_config.json
    └── tokenizer.model
```
----------------------------

#  Automated Inference Script



# [Phase 1] Model Compilation

If you do not plan to compile the model yourself, you can directly download the pre-compiled model:
```bash
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/glm4-9b_int4_seq2048_1dev.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/glm4-9b_int4_seq8192_1dev.bmodel
```
## Notes
* Model compilation must be done inside docker; it cannot be done outside docker.

### Step 1: Download docker

Download docker and start the container as follows:

```bash
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
* PS: This repo `LLM-TPU` needs to be in the current directory.

### Step 2: Download the TPU-MLIR code and install it

``` shell
pip3 install dfss  --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/tpu-mlir.tar.gz
tar zxvf tpu-mlir.tar.gz
cd tpu-mlir_v1.8.beta.0-134-g859a6f517-20240801
source ./envsetup.sh
cd ..
```
* PS: When you re-enter the docker environment and need to compile the model, you must run the above `source ./envsetup.sh` under this path before you can proceed with subsequent model compilation.

### Step 3: Model download
The ChatGLM4 model is open source for commercial use and can be downloaded from the HuggingFace official website: [glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat).
If you cannot download it from the official website, a pre-downloaded archive is also provided here.
```bash
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/glm-4-9b-chat-torch.zip
unzip glm-4-9b-chat-torch.zip
```

After downloading the official `glm-4-9b-chat` repository, you also need to set the `ChatGLM4_PATH` environment variable, which is used when exporting the model.
```bash
export ChatGLM4_PATH=$PWD/glm-4-9b-chat
```

### Step 4: Align the model environment

```bash
sudo apt-get update
sudo apt-get install pybind11-dev
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cp ./compile/files/glm-4-9b-chat/modeling_chatglm.py $ChatGLM4_PATH
cp ./compile/files/glm-4-9b-chat/config.json $ChatGLM4_PATH
```

### Step 5: Generate the onnx files

```bash
cd compile
python export_onnx.py --model_path $ChatGLM4_PATH --seq_length 512
```
* PS: By default, a model with a sequence length of 512 is exported. To export models of other lengths, you also need to modify the `seq_length` parameter in `$ChatGLM4_PATH/config.json` accordingly.

### Step 6: Generate the bmodel file

Generate a single-chip model

```bash
./compile.sh --mode int4 --name glm4-9b --seq_length 512 --addr_mode io_alone
```
Generate a W8A16 quantized model
```bash
./compile.sh --mode int8 --name glm4-9b --seq_length 512 --addr_mode io_alone
```
Generate a model with length 8192
```bash
./compile.sh --mode int8 --name glm4-9b --seq_length 8192 --addr_mode io_alone
```


<!-- Generate a dual-chip model

```bash
./compile.sh --mode int4 --num_device 2 --name glm4-9b --seq_length 512 # same as int8
``` -->

* PS1: Generating the bmodel takes roughly 3 hours or more; 64 GB of memory and more than 200 GB of disk space are recommended, otherwise OOM or "no space left" is very likely;
* PS2: If you want to compile glm4-9b, --name must be glm4-9b.
<!-- * PS3: The currently provided lib_pcie and lib_soc only contain single-chip dynamic libraries; the multi-chip part will be updated later. -->

----------------------------

# Phase 2: Executable Generation

## Compile the program (Python Demo version)
Run the following compilation (the PCIE version is the same as the SoC version):

```bash
cd python_demo
mkdir build && cd build
cmake ..
make
cp *chat* ..
```

## Model inference (Python Demo version)
```bash
cd ./python_demo
python3 pipeline.py --model_path glm4-9b_int4_1dev.bmodel --tokenizer_path ../token_config --devid your_devid
```
Other available parameters can be viewed through `pipeline.py` or by running the following command.
```bash
python3 pipeline.py --help
```

## web demo
```bash
python3 web_demo.py --model_path glm4-9b_int4_1dev.bmodel --tokenizer_path ../token_config --devid 0
```

### modeling_chatglm.py code modifications

#### Modification 1: Modify the rotary position embedding
Original code:
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

Modified code:
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

* `@torch.jit.script` is commented out to make it easier to set breakpoints
* The change to `b, sq, nq, hn = x.size(0), x.size(1), x.size(2), x.size(3)` is because the dimensions were changed to [batch_size, seq_length, head_num, head_dim] earlier; this needs to adapt to the earlier modification
* The reasons for the other modifications are the same as the previous point

#### Modification 2: pytorch_major_version

Original code:
```python
if pytorch_major_version >= 2:
```

Modified code:
```python
if False:
```

* This avoids going through flash_attention; flash_attention would later undergo graph matching in tpu-mlir and finally be matched to FAttentionOP, but FAttention cannot be used here

#### Modification 3: softmax part

Original code:
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

Modified code:
```python
matmul_result = torch.matmul(query_layer.transpose(1,2), key_layer.transpose(1, 2).transpose(2,3))
attention_scores = matmul_result * (1.0 / self.norm_factor)

```

* The modification here is very important
* `matmul_input_buffer = torch.empty` is removed because it causes a `>2G` bug when converting to onnx
* The other modifications are because the input becomes [batch_size, seq_length, head_num, head_dim], so the computation must be modified to match this shape

#### Modification 4: The extreme-value part of attention_mask

Original code:
```python
attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
```

Modified code:
```python
Comment it out directly
```

* Summation is used instead of masked_fill, because the chip backend does not support masked_fill well



#### Modification 5: (QK)*K

Original code:
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

Modified code:
```python
context_layer = torch.matmul(attention_probs, value_layer.transpose(1, 2))
```

* Because the input becomes [batch_size, seq_length, head_num, head_dim], the computation must be modified to match this shape

#### Modification 6: Processing of the QKV input and the output

The original code is located at lines 369~411
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

The modified code is located at lines 394~444
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

* Because the input becomes [batch_size, seq_length, head_num, head_dim], the computation must be modified to match this shape
* In the previous structure, there was a permute operator before and after the QKV proj, three permutes in total, but this differs from the Qwen series structure, where the three permutes are before and after FAttention. Due to this difference, it cannot be matched to FAttentionOp during lowering, so the permute of the QKV proj needs to be sunk down so that its structure stays consistent with the Qwen series structure
* In the decode stage, the output shape is changed from [batch_size, seq_length, head_num, head_dim] to [batch_size, 1, head_num, head_dim] to avoid ConcatOp and data movement
