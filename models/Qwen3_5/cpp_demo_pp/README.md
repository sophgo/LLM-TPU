# Qwen3_5 cpp_demo_pp (Pipeline Parallel)

This directory provides a **Pipeline Parallel (PP)** C++ deployment demo for Qwen3.5 on BM1684X / BM1688.
Compared with `cpp_demo` (whole model → single bmodel → single device), `cpp_demo_pp` splits the model
by component into several bmodels and loads them onto different devices, achieving multi-card pipeline
parallelism and thus supporting large models that cannot fit on a single card.

| Component | bmodel filename substring | Contents |
| --- | --- | --- |
| EmbedVit | `embed_vit` | ViT + token embedding + embedding_cache |
| Blocks | `block_NN` (zero-padded) | A contiguous segment of transformer layers (block + block_cache) |
| LmHead | `lmhead` | lm_head + greedy / sample / penalty heads |

The C++ side automatically identifies components by filename substring, with the priority
`embed_vit` → `lmhead` → `block`. Block files are sorted lexicographically, so the `_pp_combine`
output uses zero-padded numbering (`block_00`, `block_01` …). Inside each block bmodel, the original
global layer numbers are retained (`block_0`, `block_cache_0`, …), and the `Block` class automatically
parses the layer range `[start_idx, end_idx]` it carries from the network names.

## Pre-compiled Models

``` shell
# Qwen3.5-4B, six chips
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_6dev_dynamic_20260429_153532_block_00.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_6dev_dynamic_20260429_153532_block_01.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_6dev_dynamic_20260429_153532_block_02.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_6dev_dynamic_20260429_153532_block_03.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_6dev_dynamic_20260429_153532_embed_vit.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-4b-int4-autoround_w4bf16_seq2048_bm1684x_6dev_dynamic_20260429_153532_lmhead.bmodel

# Qwen3.5-9B, six chips
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-9b-int4-autoround_w4bf16_seq2048_bm1684x_6dev_dynamic_20260429_152927_block_00.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-9b-int4-autoround_w4bf16_seq2048_bm1684x_6dev_dynamic_20260429_152927_block_01.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-9b-int4-autoround_w4bf16_seq2048_bm1684x_6dev_dynamic_20260429_152927_block_02.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-9b-int4-autoround_w4bf16_seq2048_bm1684x_6dev_dynamic_20260429_152927_block_03.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-9b-int4-autoround_w4bf16_seq2048_bm1684x_6dev_dynamic_20260429_152927_embed_vit.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-9b-int4-autoround_w4bf16_seq2048_bm1684x_6dev_dynamic_20260429_152927_lmhead.bmodel

# Qwen3.5-35B-A3B, seven chips (2K)
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5_35b_a3b_7dev/qwen3.5-35b-a3b-int4-autoround_w4bf16_seq2048_bm1684x_7dev_dynamic_20260611_174448_block_00.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5_35b_a3b_7dev/qwen3.5-35b-a3b-int4-autoround_w4bf16_seq2048_bm1684x_7dev_dynamic_20260611_174448_block_01.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5_35b_a3b_7dev/qwen3.5-35b-a3b-int4-autoround_w4bf16_seq2048_bm1684x_7dev_dynamic_20260611_174448_block_02.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5_35b_a3b_7dev/qwen3.5-35b-a3b-int4-autoround_w4bf16_seq2048_bm1684x_7dev_dynamic_20260611_174448_block_03.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5_35b_a3b_7dev/qwen3.5-35b-a3b-int4-autoround_w4bf16_seq2048_bm1684x_7dev_dynamic_20260611_174448_block_04.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5_35b_a3b_7dev/qwen3.5-35b-a3b-int4-autoround_w4bf16_seq2048_bm1684x_7dev_dynamic_20260611_174448_embed_vit.bmodel
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5_35b_a3b_7dev/qwen3.5-35b-a3b-int4-autoround_w4bf16_seq2048_bm1684x_7dev_dynamic_20260611_174448_lmhead.bmodel

# Qwen3.5-35B-A3B, seven chips (10K)
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5_35b_a3b_int4_10k_7dev.tar
```

## Directory Structure

```
cpp_demo_pp/
├── CMakeLists.txt
├── pipeline.cpp           # CLI entry (same interface as cpp_demo/pipeline.cpp, with multi-card -d added)
├── chat.{hpp,cpp}         # Qwen3_5 top-level wrapper: scans the bmodel directory, assigns devices, chains the pipeline
├── embed_vit.{hpp,cpp}    # ViT + word embedding
├── block.{hpp,cpp}        # A contiguous segment of transformer layers; FA / non-FA split at FA_INTERVAL=4
├── lmhead.{hpp,cpp}       # LM Head + sampling heads
├── support.{hpp,cpp}      # common utilities (empty / init_tensors / net_launch / d2d ...)
├── include/               # tokenizers-cpp, image/video processing headers
├── lib_pcie/  lib_soc/    # libsophon static libraries (PCIe / SoC)
└── test.jpg test.mp4 gettysburg.jpg
```

## 1. Generate the PP bmodels

Use `llm_convert.py` from `tpu-mlir` with `--distribute_strategy pp --num_device N`.
`--num_device` is the total number of devices for PP (= EmbedVit 1 + Blocks K + LmHead 1).

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

The 4 generated bmodels (4-device example):

```
qwen3.5_pp_test/
├── qwen3.5-9b-int4-autoround_..._4dev_dynamic_<ts>_embed_vit.bmodel
├── qwen3.5-9b-int4-autoround_..._4dev_dynamic_<ts>_block_00.bmodel
├── qwen3.5-9b-int4-autoround_..._4dev_dynamic_<ts>_block_01.bmodel
└── qwen3.5-9b-int4-autoround_..._4dev_dynamic_<ts>_lmhead.bmodel
```

> ⚠️ Before `_pp_combine`, `llm_convert.py` also outputs the unsplit whole bmodel
> (`..._<ts>.bmodel`); do **not** put it in the directory specified by `-m`, otherwise it will be
> ignored but still waste disk space.

Constraints:
- Each block bmodel must contain at least one Full-Attention layer (Qwen3.5 has one every 4 layers),
  i.e. `K ≤ NUM_LAYERS / FA_INTERVAL` is required (the 9B model has 32 layers in total, so at most 8 PP blocks).
- The component bmodels must be generated by the same run of `llm_convert.py`, with identical timestamps.

## 2. Compile the demo

Dependencies: `cmake`, `g++`, OpenCV 4.x, libsophon.

```bash
# system opencv (apt)
sudo apt update && sudo apt install -y libopencv-dev

cd cpp_demo_pp
mkdir -p build && cd build
cmake -DTARGET_ARCH=pcie ..   # SoC side: -DTARGET_ARCH=soc
make -j8
```

If you use `/opt/sophon/sophon-opencv-latest`, change `set(SOPHON_OPENCV FALSE)` to `TRUE` in
`CMakeLists.txt` as described in `cpp_demo/README.md` before running cmake.

## 3. Run

```bash
export LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH

# single card (for debugging only; the 9B model does not actually fit)
./pipeline -m /path/to/qwen3.5_pp_test -c /path/to/Qwen3_5/config

# multi-card: assign in the order EmbedVit, Block_0, ..., Block_{K-1}, LmHead
./pipeline -m /path/to/qwen3.5_pp_test -c /path/to/Qwen3_5/config -d 8,9,10,11
```

`-c` points to the directory containing `tokenizer.json`, `preprocessor_config.json`, etc.
(see `models/Qwen3_5/config/`).

`-d` accepts a comma-separated list of device ids:

| Number of IDs | Behavior |
| --- | --- |
| 1 | All components are loaded onto that device |
| = total number of components (1+K+1) | One-to-one mapping in order |
| Other | round-robin |

The remaining CLI parameters (`-s` sampling, `-r` video ratio, `-f` video fps, etc.) are the same as
in `cpp_demo/pipeline.cpp`.

### Long Text / Chunk Prefill

When the bmodel is exported with `--use_block_with_kv` (i.e. with history support),
`cpp_demo_pp` supports **chunk prefill**: when the input exceeds the length of a single prefill chunk
(`prefill_chunk_length`, defaulting to `seq_length / 4`), it is automatically processed in segments,
thus supporting long-text input far beyond `MAX_INPUT_LENGTH`.

If a `block_prompt_<idx>` network is detected at startup, `History Support: True` is printed.

| Parameter | Description |
| --- | --- |
| `-p, --prompt` | Run a single inference in programmatic mode with the given prompt text |
| `-t, --prompt_file` | Read content from a text file as the prompt; concatenated when used together with `-p` |
| `-i, --media_path` | Specify image/video paths (comma-separated for multiple) |

Example:

```bash
# read long text from a file and ask a question
./pipeline -m /path/to/qwen3.5_kv -c config -d 8,9,10,11 \
    -t novel.txt -p "what is it talking about ?"

# pass in only the prompt text
./pipeline -m /path/to/qwen3.5_kv -c config -d 8,9,10,11 \
    -p "请介绍一下杭州"
```

In programmatic mode, `Total Tokens: <n>` is printed to indicate the number of input tokens.

At startup, the component → device mapping is printed, for example:

```
=== Multi-Device Configuration ===
EmbedVit  -> Device 8  [..._embed_vit.bmodel]
Block[0]  -> Device 9  [..._block_00.bmodel]
Block[1]  -> Device 10 [..._block_01.bmodel]
LmHead    -> Device 11 [..._lmhead.bmodel]
==================================
```

## 4. Performance Reference

Qwen3.5-9B int4 (`w4bf16, seq2048`), 4 BM1684X PCIe cards, 4-device PP:

| Metric | Value |
| --- | --- |
| First-Token Latency (FTL) | ≈ 0.25 s |
| Tokens / s (TPS) | ≈ 7.7 tok/s |

## 5. FA / Non-FA Layer Notes

Qwen3.5 inserts one Full-Attention layer (with KV cache) every `FA_INTERVAL=4` layers;
the remaining layers use linear / recurrent structures (reusing `input_mems[1]/[2]`
as conv state / recurrent state in the cache networks). `Block::is_FA(global_idx)` determines the
layer type using the global layer number, so the original global numbering of each layer must be
preserved when splitting for PP.
Each Block bmodel must contain at least one FA layer, otherwise `Block` cannot infer the KV shape.

## 6. FAQ

| Symptom | Troubleshooting |
| --- | --- |
| `bmodel not found for component …` | The filename lacks the `embed_vit` / `block` / `lmhead` substring; regenerate with the latest `_pp_combine` |
| `bmrt_load_bmodel ... NOT_INITIALIZED` | The card specified by `-d` is occupied or in Fault state; check with `bm-smi` |
| `History Support: False` (informational message) | The current `_pp_combine` output does not include history-with-kv, which is normal; long text requires a bmodel with KV |
| Long-text input reports `exceed maximum length` | Only bmodels exported with `--use_block_with_kv` + `--prefill_chunk_length` support chunk prefill |
| Cannot find `libbmrt.so` / `libbmlib.so` | `export LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH` |
| Stuck during the loading phase | The card is in Fault state; switch to another set of device ids or restart the driver |
