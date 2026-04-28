# Qwen3_5 cpp_demo_pp (Pipeline Parallel)

本目录提供 Qwen3.5 在 BM1684X / BM1688 上的 **Pipeline Parallel (PP)** C++ 部署 demo。
相较于 `cpp_demo`（整模型 → 单 bmodel → 单 device），`cpp_demo_pp` 将模型按组件
拆分为若干 bmodel，分别加载到不同 device，实现多卡流水并行，从而支持单卡放不下的大模型。

| 组件 | bmodel 文件名子串 | 内容 |
| --- | --- | --- |
| EmbedVit | `embed_vit` | ViT + token embedding + embedding_cache |
| Blocks | `block_NN` (零填充) | 一段连续的 transformer 层 (block + block_cache) |
| LmHead | `lmhead` | lm_head + greedy / sample / penalty heads |

C++ 端通过文件名子串自动识别组件：优先级 `embed_vit` → `lmhead` → `block`，
block 文件按字典序排序，因此 `_pp_combine` 输出使用零填充编号 (`block_00`、`block_01` …)。
每个 block bmodel 内部沿用原始全局层号 (`block_0`、`block_cache_0`、…)，
`Block` 类自动从 network names 解析其承载的层范围 `[start_idx, end_idx]`。

## 目录结构

```
cpp_demo_pp/
├── CMakeLists.txt
├── pipeline.cpp           # CLI 入口（与 cpp_demo/pipeline.cpp 接口一致，新增多卡 -d）
├── chat.{hpp,cpp}         # Qwen3_5 顶层封装：扫描 bmodel 目录、分配 device、串联流水
├── embed_vit.{hpp,cpp}    # ViT + 词嵌入
├── block.{hpp,cpp}        # 连续若干 transformer 层；FA_INTERVAL=4 的 FA / 非 FA 二分
├── lmhead.{hpp,cpp}       # LM Head + 采样头
├── support.{hpp,cpp}      # 公共工具 (empty / init_tensors / net_launch / d2d ...)
├── include/               # tokenizers-cpp、图像/视频处理头文件
├── lib_pcie/  lib_soc/    # libsophon 静态库 (PCIe / SoC)
└── test.jpg test.mp4 gettysburg.jpg
```

## 1. 生成 PP bmodel

使用 `tpu-mlir` 的 `llm_convert.py`，加 `--distribute_strategy pp --num_device N`。
`--num_device` 即 PP 的总 device 数（= EmbedVit 1 + Blocks K + LmHead 1）。

```bash
source /path/to/tpu-mlir/envsetup.sh

llm_convert.py \
    -m /path/to/Qwen3.5-9B-int4-AutoRound \
    -s 2048 \
    --max_input_length 1024 \
    --num_device 4 \
    --distribute_strategy pp \
    -o qwen3.5_pp_test
```

生成的 4 个 bmodel（4 device 示例）：

```
qwen3.5_pp_test/
├── qwen3.5-9b-int4-autoround_..._4dev_dynamic_<ts>_embed_vit.bmodel
├── qwen3.5-9b-int4-autoround_..._4dev_dynamic_<ts>_block_00.bmodel
├── qwen3.5-9b-int4-autoround_..._4dev_dynamic_<ts>_block_01.bmodel
└── qwen3.5-9b-int4-autoround_..._4dev_dynamic_<ts>_lmhead.bmodel
```

> ⚠️ `llm_convert.py` 在 `_pp_combine` 之前还会输出未拆分的整体 bmodel
> (`..._<ts>.bmodel`)，**不要**把它放进 `-m` 指定的目录，否则会被忽略但白占磁盘。

约束：
- 每个 block bmodel 至少包含一层 Full-Attention 层（Qwen3.5 每 4 层一次），
  即要求 `K ≤ NUM_LAYERS / FA_INTERVAL`（9B 共 32 层，PP block 数最多 8）。
- 各组件 bmodel 必须由同一次 `llm_convert.py` 生成，时间戳一致。

## 2. 编译 demo

依赖：`cmake`、`g++`、OpenCV 4.x、libsophon。

```bash
# 系统 opencv (apt)
sudo apt update && sudo apt install -y libopencv-dev

cd cpp_demo_pp
mkdir -p build && cd build
cmake -DTARGET_ARCH=pcie ..   # SoC 端：-DTARGET_ARCH=soc
make -j8
```

如使用 `/opt/sophon/sophon-opencv-latest`，按 `cpp_demo/README.md` 所述将
`CMakeLists.txt` 中 `set(SOPHON_OPENCV FALSE)` 改为 `TRUE` 后再 cmake。

## 3. 运行

```bash
export LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH

# 单卡（仅用于调试，9B 模型实际放不下）
./pipeline -m /path/to/qwen3.5_pp_test -c /path/to/Qwen3_5/config

# 多卡：按 EmbedVit, Block_0, ..., Block_{K-1}, LmHead 顺序分配
./pipeline -m /path/to/qwen3.5_pp_test -c /path/to/Qwen3_5/config -d 8,9,10,11
```

`-c` 指向包含 `tokenizer.json`、`preprocessor_config.json` 等的目录
（参见 `models/Qwen3_5/config/`）。

`-d` 接受逗号分隔的 device id 列表：

| ID 数量 | 行为 |
| --- | --- |
| 1 | 全部组件加载到该 device |
| = 组件总数 (1+K+1) | 按顺序一一对应 |
| 其它 | round-robin |

其余 CLI 参数 (`-s` 采样、`-r` 视频比例、`-f` 视频 fps 等) 与
`cpp_demo/pipeline.cpp` 一致。

启动时会打印组件 → device 映射，例如：

```
=== Multi-Device Configuration ===
EmbedVit  -> Device 8  [..._embed_vit.bmodel]
Block[0]  -> Device 9  [..._block_00.bmodel]
Block[1]  -> Device 10 [..._block_01.bmodel]
LmHead    -> Device 11 [..._lmhead.bmodel]
==================================
```

## 4. 性能参考

Qwen3.5-9B int4 (`w4bf16, seq2048`)，4 张 BM1684X PCIe，4-device PP：

| 指标 | 数值 |
| --- | --- |
| First-Token Latency (FTL) | ≈ 0.25 s |
| Tokens / s (TPS) | ≈ 7.7 tok/s |

## 5. FA / 非 FA 层说明

Qwen3.5 每 `FA_INTERVAL=4` 层插入一层 Full-Attention（带 KV cache），
其余层为 linear / recurrent 结构（在 cache 网络中复用 `input_mems[1]/[2]`
作为 conv state / recurrent state）。`Block::is_FA(global_idx)` 用全局
层号判断层类型，因此 PP 拆分时务必保留每层原始的全局编号。
每个 Block bmodel 至少需包含一层 FA 层，否则 `Block` 无法推断 KV 形状。

## 6. 常见问题

| 现象 | 排查 |
| --- | --- |
| `bmodel not found for component …` | 文件名缺少 `embed_vit` / `block` / `lmhead` 子串；用最新版 `_pp_combine` 重新生成 |
| `bmrt_load_bmodel ... NOT_INITIALIZED` | `-d` 指定的卡被占用或处于 Fault；`bm-smi` 检查 |
| `History Support: False`（信息提示） | 当前 `_pp_combine` 输出不带 history-with-kv，正常 |
| 调用 `forward_first_with_kv` 抛异常 | 该路径暂未实现；不要使用 history-enabled 的 bmodel |
| 找不到 `libbmrt.so` / `libbmlib.so` | `export LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH` |
| 加载阶段卡住 | 卡处于 Fault；换一组 device id 或重启驱动 |
