# Qwen3_VL 分布式 Pipeline 推理

将 Qwen3_VL 推理拆分为三个独立程序，分别运行在三个芯片上，通过 TCP 网络传输中间数据。

## 架构

```
Step0 (embed+vit)  --TCP-->  Step1 (block/block_cache)  --TCP-->  Step2 (lm_head)
       ^                                                               |
       |___________________________ token result ______________________|
```

| 程序 | 运行网络 | 功能 |
|------|---------|------|
| `step0_embed_vit` | embedding, embedding_cache, vit | 用户交互、分词、embedding和VIT推理，发送hidden states给Step1 |
| `step1_block` | block_0..N, block_cache_0..N, add | 接收hidden states，运行Transformer blocks，管理KV cache，发送输出给Step2 |
| `step2_lmhead` | lm_head, greedy_head/sample_head | 接收hidden state，运行LMHead生成token，返回给Step1→Step0 |

## 数据流

### Prefill 阶段
1. **Step0**: 运行 embedding + VIT → 将 hidden_states + position_ids + deepstack 发送给 Step1
2. **Step1**: 接收数据 → 运行所有 block 层 → 提取最后一个token的hidden state → 发送给 Step2
3. **Step2**: 接收 hidden state → 运行 lm_head → 生成 token → 返回给 Step1 → 转发给 Step0

### Decode 阶段
1. **Step0**: 运行 embedding_cache → 将输出 hidden_state + position_ids 发送给 Step1
2. **Step1**: 接收数据 → 运行所有 block_cache 层 → 发送输出给 Step2
3. **Step2**: 接收 hidden state → 运行 lm_head → 生成 token → 返回

## 编译

```bash
cd distributed
mkdir build && cd build
cmake .. -DTARGET_ARCH=pcie
make -j
```

## 运行

需要为三个程序分别准备各自的 bmodel（包含对应的网络）。

### 启动顺序

**必须按 Step2 → Step1 → Step0 的顺序启动**（因为前者需要先监听端口）。

```bash
# 终端1 - Step2 (lmhead，监听端口10002)
./step2_lmhead -m /path/to/lmhead.bmodel -d 2 -p 10002

# 终端2 - Step1 (block，监听端口10001，连接Step2)
./step1_block -m /path/to/block.bmodel -d 1 -p 10001 \
    --step2_host 127.0.0.1 --step2_port 10002

# 终端3 - Step0 (embed+vit，连接Step1)
./step0_embed_vit -m /path/to/embed_vit.bmodel -c /path/to/config -d 0 \
    --step1_host 127.0.0.1 --step1_port 10001
```

### 参数说明

**step0_embed_vit:**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model` | Embed+VIT bmodel 路径 | 必须 |
| `-c, --config` | 配置目录（含 tokenizer.json） | 必须 |
| `-d, --devid` | 设备ID | 0 |
| `-r, --video_ratio` | 视频缩放比例 | 0.25 |
| `-f, --video_fps` | 视频帧率 | 1.0 |
| `--step1_host` | Step1 IP 地址 | 127.0.0.1 |
| `--step1_port` | Step1 端口 | 10001 |

**step1_block:**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model` | Block bmodel 路径 | 必须 |
| `-d, --devid` | 设备ID | 0 |
| `-p, --port` | 监听端口 | 10001 |
| `--step2_host` | Step2 IP 地址 | 127.0.0.1 |
| `--step2_port` | Step2 端口 | 10002 |

**step2_lmhead:**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model` | LMHead bmodel 路径 | 必须 |
| `-c, --config` | 配置目录（含 generation_config.json） | 可选 |
| `-d, --devid` | 设备ID | 0 |
| `-p, --port` | 监听端口 | 10002 |
| `-s, --do_sample` | 启用采样 | 关闭 |

### 跨机器部署

如果三个芯片在不同机器上，将 `--step1_host` 和 `--step2_host` 改为对应机器的 IP 地址即可。

## bmodel 拆分

需要将原始的完整 bmodel 按网络名称拆分为三个独立的 bmodel：

- **embed_vit.bmodel**: 包含 `embedding`, `embedding_cache`, `vit` 网络
- **block.bmodel**: 包含 `block_0` ~ `block_N`, `block_cache_0` ~ `block_cache_N`, `add` 网络
- **lmhead.bmodel**: 包含 `lm_head`, `greedy_head` (可选), `sample_head` (可选) 网络
