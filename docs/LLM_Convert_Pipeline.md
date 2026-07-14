# LLM-TPU ↔ tpu-mlir Converter Internals

This document explains how the demos in this repository relate to the model
compiler in [tpu-mlir](https://github.com/sophgo/tpu-mlir) `python/llm/`. Read
it when you need to add a model, change a compile-time feature, or debug a
mismatch between a compiled bmodel and a demo's `chat.cpp`.

## The two halves of the pipeline

```
tpu-mlir (compiler repo)                        LLM-TPU (this repo, demos/runtime)
─────────────────────────                       ──────────────────────────────────
python/tools/llm_convert.py  ──┐
  └─ python/llm/*Converter.py  │  HuggingFace/GGUF weights
     LlmInfo / ModelHandle     ▼
                          model_deploy.py per piece
                          model_tool --combine
                                   │
                                   ▼
                     <name>.bmodel  +  config/ dir
                                   │
                                   ▼
                     models/<Model>/python_demo/pipeline.py  (tokenize, chat template)
                                    └─ chat.cpp (pybind11 → bmrt: prefill/decode/KV-cache/sample)
```

- **tpu-mlir `python/llm/`** turns source weights into a quantized `.bmodel`
  plus a runtime `config/` directory. It runs on an x86 host inside the
  `sophgo/tpuc_dev` Docker image; no TPU is needed for compilation.
- **LLM-TPU `models/<Model>/`** consumes exactly those two artifacts. The demo
  never reads the original HuggingFace weights — only the bmodel (via bmrt in
  `chat.cpp`) and the `config/` dir (tokenizer / chat template / generation
  config via `transformers` in `pipeline.py`).

## Compiler entry point: `llm_convert.py`

`tpu-mlir/python/tools/llm_convert.py` parses arguments, loads the model
config, and dispatches to a converter class via the `LLM_CONVERTERS` table —
a list of `(model_types, module, class, options)` entries. The dispatch key is
`config.json`'s `model_type` for safetensors models, or the GGUF
`general.architecture` field (mapped through
`GGUFModelHandle.ARCH_TO_MODEL_TYPE`) for `.gguf` files. Adding support for a
new architecture is one line in this table plus a converter class.

Per-model options in the table:

- `default_max_shape` — fallback `--max_pixels` for the ViT (e.g. 672×896 for
  qwen2_5_vl, 980×980 for minicpmv, 768×768 otherwise).
- `pixel_multiple` — `max_pixels` must be a multiple of `m*m` (28 for Qwen2-VL
  family, 32 for qwen3_vl, 56 for minicpmv4_6, 16 for falcon_perception).
- `force_dynamic` — flips `--dynamic` on automatically (qwen3_5, minicpmv4_6).

Argument normalization done by the entry point before the converter runs:

- `--use_history_kv` forces `--dynamic`, defaults `--chunk_length` to
  `seq_length // 4`, and **derives** `--max_input_length = chunk_length`
  (passing `--max_input_length` explicitly is an error).
- `--chunk_length` must be ≤ `seq_length // 2`.
- `-o/--out_dir` defaults to `./<model_name>_<chip>_<quantize>`.
- `--dry_run` prints the resolved configuration without converting.

### model_type → converter → demo mapping

| `model_type` (config.json) | Converter class (tpu-mlir `python/llm/`) | Demo dir(s) in this repo |
|---|---|---|
| `qwen3`, `qwen2`, `llama`, `minicpm`, `qwen2_moe` | `LlmConverter` (generic) | Qwen3, Qwen2, Qwen2_5, Qwen1_5, Llama2/3, MiniCPM3/4, DeepSeek-V2, Yi, Mistral, Baichuan2, ... |
| `mllama` | `Llama3_2VConverter` | Llama3_2-Vision |
| `chatglm` | `Chatglm3Converter` | ChatGLM3, GLM4 |
| `phi3` | `Phi3Converter` | Phi-3 |
| `qwen2_vl` | `Qwen2VLConverter` | Qwen2_VL |
| `qwen2_5_vl` | `Qwen2_5VLConverter` | Qwen2_5_VL |
| `qwen3_vl` | `Qwen3VLConverter` | Qwen3_VL |
| `qwen3_5`, `qwen3_5_moe` | `Qwen3_5Converter` | Qwen3_5 |
| `qwen2_5_omni` | `Qwen2_5OConverter` | Qwen2_5_Omni |
| `qwen3_asr` | `Qwen3AsrConverter` | Qwen3_ASR |
| `internvl_chat` | `InternVL3Converter` | InternVL2, InternVL3 |
| `gemma3` | `Gemma3Converter` | Gemma3 |
| `gemma4` | `Gemma4Converter` | Gemma4 |
| `glm4v` | `GLM4VConverter` | GLM4V |
| `minicpmv` | `MiniCPMV4Converter` | MiniCPMV4, MiniCPM-V-2_6 |
| `minicpmv4_6` | `MiniCPMV4_6Converter` | MiniCPMV4_6 |
| `janus` | `JanusConverter` | Janus-Pro |
| `paddleocr_vl` | `PaddleOCRVLConverter` | — |
| `lfm2_vl` | `LFM2VLConverter` | — |
| `locateanything` | `LocateAnythingConverter` | LocateAnything |
| `falcon_perception` | `FalconPerceptionConverter` | Falcon-Perception |

Note: many demos in `models/` share a converter because they share an
architecture (e.g. Qwen2.5 LLMs are `model_type: qwen2` → `LlmConverter`).
When in doubt, check the source model's `config.json` `model_type`.

## Converter internals (`python/llm/`)

### `LlmConverter` (base class for all converters)

`run()` executes four steps:

1. **`gen_config()`** — build the runtime `config/` dir (see below).
2. **`gen_all_mlir()`** — emit one MLIR module per network piece, in parallel
   threads: `vit` (VLMs), `embedding` + `embedding_cache` + `lm_head`
   (+ lora / sample heads), then `block_<i>` / `block_cache_<i>` /
   `block_kv_<i>` for every transformer layer. Each module is sanity-checked
   with `tpuc-opt --shape-infer`.
3. **`compile_all()`** — queue one `model_deploy.py` per piece (parallel
   under `<piece>.log`) with the right `--quantize`, `--q_group_size`,
   `--num_device/--num_core`, `--addr_mode`, `--dynamic` flags.
4. **`combine()`** — `model_tool --combine` merges all per-piece bmodels into
   the final single `.bmodel`. With `--distribute_strategy pp`, pieces are
   instead grouped into `{base}_embed_vit`, `{base}_block_XX`,
   `{base}_lmhead` bmodels packed in a `{base}_pp.tar` (consumed by
   `cpp_demo_pp`).

Output naming: the bmodel dir is
`<model>_<quantize>_seq<N>_<chip>_<n>dev[_<b>b][_history]_{dynamic|static}`
(bm1688/cv186x use `_<n>core` instead of `_<n>dev`), and the final bmodel
appends a timestamp. `--again` resumes an interrupted conversion by skipping
pieces whose outputs already exist.

### Weight-name mapping: `LlmInfo.py`

Converters do not hardcode HF weight paths. `LlmInfo.py` defines:

- `LlmList` — logical roles (`LAYERS`, `EMBEDING`, `Q_PROJ`, `K_NORM`,
  `MLP_GATE`, `EXPERTS_UP`, `LMHEAD`, ...).
- `ModelConfig` — maps config.json field names (e.g. ChatGLM uses
  `ffn_hidden_size`, `multi_query_group_num`).
- `ModelInfo` — maps each role to the model's actual weight path.
  `COMMON_INFO` covers Llama/Qwen-style models; there are variants for
  ChatGLM3, Phi-3 (fused `qkv_proj` / `gate_up_proj`), Gemma3/4, Janus,
  mllama (cross-attention), Qwen2.5-Omni/Qwen3-ASR (`thinker.*`), GLM4V,
  MiniCPMV, Qwen3-VL, LFM2, and Falcon-Perception (fully custom fused block).

A converter picks its `ModelInfo` and then refers to weights only by role.

### Weight loading: `ModelHandle.py`

Two loader classes, selected by the input format:

- **`SafetensorsModelHandle`** (HF directory): reads tensors via `LlmLoad`.
  `init_quantization()` inspects `config.json`'s `quantization_config`:
  - `gptq` / `awq` / `compressed-tensors` / `auto-round` (mapped to gptq/awq
    by packing format) → the bmodel quantize is **forced** from the source
    (`bits` + dtype); `-q auto` resolves automatically. AWQ requires
    `version == gemm`, 4 bits, and forces `w4f16`.
  - `fp8` → `Fp8MatMulOp` path with block scales.
  - no `quantization_config` → unquantized source; you must pass `-q`
    (`bf16`/`f16`/`w8bf16`/`w4bf16`/...) and it must match the model dtype.
  - `gen_config()` copies the HF directory into `config/`, excluding weight
    files (`*.safetensors`, `*.bin`, `*.pt`, ...) and `*.py`.
- **`GGUFModelHandle`** (single `.gguf` file): per-tensor quant types are
  converted through `QuantConverter`; mixed-quant fallbacks are tracked per
  layer (`_block_quant_info`) so `model_deploy.py` gets per-block quantize
  args. For VLM GGUFs, an `mmproj*.gguf` must sit next to the model (or be
  passed via `--mmproj`); its vision tensors are merged in. `gen_config()`
  **synthesizes** `tokenizer.json` / `tokenizer_config.json` /
  `generation_config.json` etc. from GGUF metadata, so the demo side looks
  identical to the safetensors case.

## The bmodel ↔ demo contract

This is the most important section for demo work. `chat.cpp` discovers
everything by **network name** and **tensor shapes** at load time
(`bmrt_get_network_info`), so net names and I/O layouts are a stable contract
between compiler and demos.

### Networks inside a bmodel

| Net name | Count | Produced when | Role |
|---|---|---|---|
| `embedding` | 1 | always (unless `--embedding_disk`) | prefill embedding, input len = `max_input_length` |
| `embedding_cache` | 1 | always (unless `--embedding_disk`) | decode embedding, input len = 1 |
| `lm_head` | 1 | always | hidden → **token id** (TopK-1 fused, default) or → **logits** (with `--do_sample` or lora) |
| `greedy_head` | 1 | `--do_sample` (or lora) | logits → argmax token |
| `sample_head` | 1 | `--do_sample` (or lora) | penalty + temperature + top-k + top-p on TPU |
| `block_<i>` | num_layers | always | prefill layer i (full attention, produces K/V) |
| `block_cache_<i>` | num_layers | always | decode layer i (1 token, reads/writes KV cache) |
| `block_kv_<i>` | num_layers | `--use_history_kv` | prefill with history-KV concat (chunked prefill) |
| `block_cache_<i>_<s>` | per stage | `--chunk_length` > 0 | extra decode stages for multi-stage decode chunking |
| `vit` | 1 | VLM converters (`do_vit = True`) | vision encoder + projector/merger, dynamic shapes |
| `embedding_lora`, `embedding_cache_lora`, `lm_head_lora` | 1 each | `--lora_max_rank` > 0 | LoRA add-on nets (zero-init weights, filled at runtime) |
| `add*` | a few | `--distribute_strategy pp` | residual adds for pipeline-parallel splits |

Feature detection in demos (e.g. `models/Qwen3/python_demo/chat.cpp`):

- `support_prefill_kv` = `block_kv_0` exists → `--use_history_kv` build; the
  demo then keeps KV on-device across turns (`clear_kv()` instead of
  re-prefilling history).
- `lmhead_with_topk` = `lm_head` output dim is 1 → greedy id comes straight
  from `lm_head`; otherwise logits go through `greedy_head`/`sample_head`.
- `MAX_INPUT_LENGTH` = `embedding` input seq dim; `SEQLEN` = `block_cache_0`
  KV input seq dim. Demos never hardcode these.

### Net I/O layouts

`block_<i>` (prefill, static mask) inputs:

```
input_states   [1, max_input_length, hidden_size]        F32 (bf16/f16 after quant)
position_ids   [1, max_input_length]                     INT32
attention_mask [1, 1, max_input_length, max_kv_len]      F32   (absent when --dynamic: a static small mask weight is used instead)
```

outputs: `output_states`, `k_cache`, `v_cache`
(`[1, len, num_key_value_heads, head_dim]`).

`block_kv_<i>` additionally takes `history_k` / `history_v`
(`[1, seq_length, kv_heads, head_dim]`) right after the mask input, and the
mask covers `seq_length + chunk` columns.

`block_cache_<i>` (decode) inputs:

```
input_states   [batch, 1, hidden_size]
position_ids   [batch, 1]            (or [1,1] when batch == 1)
attention_mask [batch, 1, 1, kv_len(+1)]
past_k/past_v  [batch, kv_len, num_key_value_heads, head_dim]
```

outputs: `output_states` (+ updated k/v when batch == 1; batch > 1 writes the
KV cache in place via `use_insert`).

`lm_head`: input `[1, hidden_size]` → output `[1, 1]` token id (topk) or
`[1, vocab_size]` logits. `sample_head` inputs: `m_logits`, `input_ids`,
`penalty`, `temperature`, `top_k`, `top_p` → sampled distribution + indices.

`vit`: input pixel patches shaped by `--max_pixels`; output image embeddings
in LLM hidden size, which `pipeline.py` splices into the token embedding
sequence at image-placeholder positions.

## Compile flag → runtime feature cheat sheet

| `llm_convert.py` flag | Effect on bmodel | Demo-side consequence |
|---|---|---|
| `-s/--seq_length` | KV cache capacity; rotary tables sized to it | demo `SEQLEN`; history truncated/reset beyond it |
| `--max_input_length` | prefill chunk size (< seq_length) | demo `MAX_INPUT_LENGTH`; longer prompts are chunked |
| `--dynamic` | dynamic prefill stages + small static mask | demo picks stage by real length (`net_launch_dyn`) |
| `--use_history_kv` (+`--chunk_length`) | adds `block_kv_*`; forces dynamic | multi-turn reuses on-device KV (`support_prefill_kv`) |
| `--chunk_length` | decode chunk stages `block_cache_*_<s>` | multi-stage decode demos |
| `--do_sample` | logits lm_head + greedy/sample heads | `pipeline.py --do_sample` sets sampling params from `generation_config.json` |
| `--embedding_disk` | no embedding nets; `config/embedding.bin` instead | demo embeds on CPU |
| `-q/--quantize` | weight quant (`w4bf16`, `bf16`, ...); `auto` follows source quant | nothing to set at runtime; accuracy/size tradeoff at compile time |
| `-g/--q_group_size` | per-group quant granularity | — |
| `--num_device N` / `--distribute_strategy tp` | multi-chip tensor parallel | `python_demo_parallel`, `--devid 0,1,...` |
| `--distribute_strategy pp` | split bmodels in `{base}_pp.tar` | `cpp_demo_pp` |
| `-b/--batch N` | batched decode nets (`use_insert`) | multiuser demo variants |
| `--lora_max_rank R` | lora nets with zero weights | `support/tools` lora export/load flow |
| `--max_pixels W,H` | ViT input budget | VLM image preprocessing limit |
| `--only_mlir` / `--debug` | stop at MLIR / keep intermediates | analysis only |

## The `config/` directory contract

Every bmodel must be paired with the `config/` dir generated alongside it
(`gen_config()`). Demos load it with
`AutoTokenizer.from_pretrained(--config_path)` and
`GenerationConfig.from_pretrained(...)`. Contents come from the source model
(safetensors: copied verbatim minus weights; GGUF: synthesized from metadata)
— typically `config.json`, `generation_config.json`, `tokenizer.json`,
`tokenizer_config.json`, `vocab.json`/merges, `chat_template`, plus
`embedding.bin` when `--embedding_disk` was used.

Do **not** mix a bmodel with a config dir from a different source model or a
different tokenizer revision.

## Recipe: adding support for a new model

1. **Compiler side (tpu-mlir repo):**
   - If the architecture matches an existing family (standard Llama/Qwen-style
     decoder), `LlmConverter` + an appropriate `ModelInfo` in `LlmInfo.py` may
     be enough — weight paths and config field names are the only things that
     vary.
   - Otherwise subclass `LlmConverter` (see `Chatglm3Converter` for fused-QKV,
     `Qwen2_5VLConverter` for adding a `vit` net: set `self.do_vit = True`,
     implement `gen_vit_mlir`, append compile steps).
   - Register the `model_type` in `LLM_CONVERTERS` in `llm_convert.py`.
   - Verify with `--dry_run`, then `--only_mlir`, then a full compile.
2. **Demo side (this repo):**
   - Copy the closest existing model dir (same arch family, LLM vs VLM).
   - Replace `config/` with the converter-generated one.
   - Adapt `python_demo/pipeline.py` (chat template, system prompt, media
     handling) and `python_demo/chat.cpp` (class name, any model-specific net
     I/O). Keep net-name lookups identical unless the compiler side changed
     the contract.
   - Add `run_demo.sh` (dfss download URL + build + launch), `README.md`,
     and a mapping line in top-level `run.sh`.
3. **Validate:** `regression/run.sh` (compile check across all demos), then
   run on device and compare against the source model; use `support/debug/`
   helpers if outputs mismatch.

## Key files

tpu-mlir repo (`/workspace/tpu-mlir`):

- `python/tools/llm_convert.py` — CLI entry + dispatch table
- `python/llm/LlmConverter.py` — base converter (embedding/lm_head/sample
  head/block MLIR, compile + combine)
- `python/llm/LlmInfo.py` — weight-path/config-field mapping per family
- `python/llm/ModelHandle.py` — safetensors & GGUF loaders, quant detection,
  `config/` generation
- `python/llm/<Arch>Converter.py` — per-architecture subclasses
- `python/llm/QuantConverter.py`, `GGUFQuantLoad.py` — GGUF quant handling

This repo:

- `models/<Model>/python_demo/chat.cpp` — net-by-name loading, KV cache
  management, prefill/decode/sampling on bmrt
- `models/<Model>/python_demo/pipeline.py` — tokenizer, chat template,
  decode loop, EOS handling
- `harness/task/bmodel_task.py` — loads a demo's `Model` + `chat` module for
  accuracy evals
