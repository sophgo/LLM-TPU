---
name: tpu-mlir-llm-converter
description: >-
  Internals of the tpu-mlir LLM compiler (/workspace/tpu-mlir python/llm/ +
  python/tools/llm_convert.py) that produces the bmodels this repo's demos run.
  Use when adding a model, debugging a bmodel↔demo mismatch, choosing or
  verifying llm_convert.py flags, or reasoning about how a compile-time option
  (quantize, dynamic, history KV, sampling, VLM vit) changes what the demo side
  sees.
---

# tpu-mlir LLM converter internals (`/workspace/tpu-mlir`)

Pick the right reference first:

- **bmodel ↔ demo contract** (net names, I/O shapes, `config/` contents, flag →
  runtime feature table, add-a-model recipe): `docs/LLM_Convert_Pipeline.md` in
  this repo. This skill deliberately does not repeat it.
- **Compiler internals** (dispatch mechanics, arg normalization, quant
  detection, gotchas): this skill.
- **Source of truth** for any flag or table entry: the code itself, especially
  `python/tools/llm_convert.py`. Verify before repeating specifics — the repo
  evolves.

## Key files

| File | Role |
|---|---|
| `python/tools/llm_convert.py` (~330 lines) | CLI entry, arg parsing/normalization, `LLM_CONVERTERS` dispatch table |
| `python/llm/LlmConverter.py` (~2800 lines) | Base converter: MLIR gen for embedding/lm_head/sample_head/block/vit, compile, combine |
| `python/llm/LlmInfo.py` (~460 lines) | `LlmList` weight roles + per-family `ModelInfo` weight-path maps |
| `python/llm/ModelHandle.py` (~2500 lines) | `SafetensorsModelHandle` / `GGUFModelHandle`: weight loading, quant detection, `config/` generation |
| `python/llm/<Arch>Converter.py` | Per-arch subclasses (vit, fused QKV, mrope, custom blocks) |
| `python/llm/QuantConverter.py`, `GGUFQuantLoad.py`, `gguf_compat.py` | GGUF quant-type conversion/loading |
| `python/llm/transformers_compat.py` | `load_auto_config` shim over transformers versions |

## Dispatch mechanics (beyond the doc's model_type table)

`llm_convert.py` resolves `model_type`, then scans `LLM_CONVERTERS` — a list of
`(model_types, module, class, options)`; adding a model is one line here.

- **safetensors dir** → `config.json`'s `model_type` via
  `load_auto_config(..., trust_remote_code=True)`.
- **GGUF file** → `general.architecture`, mapped through
  `GGUFModelHandle.ARCH_TO_MODEL_TYPE` (`qwen2_5vl→qwen2_5_vl`, `llama3→llama`,
  `qwen35→qwen3_5`, ...). `general.tags` containing `internvl` overrides to
  `internvl_chat`. **Unknown arch falls through as-is and defaults to `qwen3`
  when the field is missing** — so llama-like GGUFs often convert via
  `LlmConverter` with no table edit.
- VLM GGUF (`VLM_ARCHS`) requires an `mmproj*.gguf`: auto-discovered next to
  the model only if exactly one match (several or none → error), or pass
  `--mmproj`. Its vision tensors merge into the main tensor map.
- Table options: `default_max_shape` (fallback `--max_pixels`),
  `pixel_multiple` m (max_pixels must be a multiple of m²), `force_dynamic`.

## Entry-point normalization (code-level extras)

- `max_input_length` falls back to `seq_length` when unset or ≥ seq_length.
- `chunk_length > 0` builds `decode_chunk_list` by doubling from chunk_length
  up to seq_length, reversed → the `block_cache_<i>_<s>` decode stages.
- `--max_pixels W,H` is split into `args.max_shape` + scalar `args.max_pixels =
  W*H` before the converter sees it.
- Deprecated: `--dynamic_vit` (vit always dynamic now), `--input_length_list`
  (use `--dynamic`).
- Paths with `qwen` + `asr` in the name trigger `import qwen_asr` before config
  load (registers the custom HF model type).

## Converter flow extras (beyond the doc's 4-step summary)

- MLIR gen runs in a `ThreadPoolExecutor` — workers from env
  `LLM_GEN_MLIR_WORKERS` (default 4). `--debug` serializes it instead.
- Every generated MLIR module is sanity-checked with `tpuc-opt --shape-infer`;
  a failure there means a bug in the converter's graph construction.
- `submit_deploy_task` builds each `model_deploy.py` command: `--addr_mode
  basic` normally, `io_alone` for cache/lora pieces; always `--high_precision`
  and `--disable_gdma_check`; `--dynamic`/`--q_symmetric` only when set
  globally. Per-piece logs land at `<bmodel_dir>/<piece>.log` — check these
  first when compile fails.
- `register_bmodel(name, with_size)`: `with_size=False` marks tied/shared
  bmodels (e.g. `embedding_cache`) whose bytes don't count toward the combined
  size sanity check (combined ≤ 1.2× sum of pieces).
- Decode-chunk stages compile in a **second pass** after the main compile.
- `--only_mlir` generates just `block_0` (+`block_3` for qwen3_5) — enough for
  `llm_analyse.py`-style inspection.

## Weight roles (`LlmInfo.py`)

Converters never hardcode HF paths. A `ModelInfo` maps `LlmList` roles to real
weight paths, and `ModelConfig` maps config.json field names (chatglm
`ffn_hidden_size`, falcon_perception `dim`/`n_layers`). Notable roles: fused
`QKV_WB`/`ATT_D` (chatglm, phi3), `MLP_GATE_UP`, MoE `GATE`/`EXPERTS_*`/
`SHARED_EXPERT_*` (paths contain a literal `expert_id` placeholder), mllama
cross-attn `C_*`, gemma4 `PER_LAYER_*` + `EMBEDING_PER_LAYER`. Supporting a
llama-like variant is often just a new `ModelInfo` — check `COMMON_INFO`
first. `tie_word_embeddings` makes lm_head reuse the embedding weight.

## Quant detection (`ModelHandle.init_quantization`)

Safetensors, from `config.json quantization_config`:

- none → unquantized source; `-q auto` is an **error** — pass `-q`, and it must
  match the model dtype (`bf16` model + `-q f16` ⇒ RuntimeError).
- `gptq` → quantize forced to `get_qtype(dtype, bits)`; `group_size`/`bits`
  taken from the config, overriding `-g`.
- `awq` → requires `version == gemm` and 4 bits; **forced to `w4f16`**.
- `auto-round` → remapped by `packing_format` to gptq (`auto_gptq`) or awq
  (`auto_awq`).
- `compressed-tensors` → only compressed pack-quantized, single config group,
  int weights.
- `fp8` → separate path using `activation_scheme`, `fmt`, `weight_block_size`.
- An explicit `-q` disagreeing with the source quant is overridden with a
  warning. `get_qtype`: fp16→w4f16/w8f16/f16, bf16→w4bf16/w8bf16/bf16;
  `half_precision_quantize` ("bf16"/"f16") covers non-quantized pieces (vit
  default, detected per vit weight bits when quantized).

GGUF: per-tensor GGML quant types converted via `QuantConverter`;
`_block_quant_info` tracks per-block quantize args and float fallbacks
(`is_block_float_fallback`, lmhead fallback) → `model_deploy.py` gets
**per-block** flags via `compile_block_args`. This exists for mixed-quant
GGUFs (e.g. some blocks q4, some q6/q8).

## Subclass patterns (adding an arch)

- LLM variant of the llama family → reuse `LlmConverter`, add a `ModelInfo`.
- Fused QKV → `Chatglm3Converter` / `Phi3Converter` (`QKV_WB`, `ATT_D`).
- VLM → set `self.do_vit = True`, implement `init_vconfig` + `gen_vit_mlir`
  (vision blocks + projector), override rotary for mrope (Qwen2_5VL), append
  vit compile steps. Vit I/O contract is in the doc.
- MoE → `moe()` + `_set_moe_expert_weights`; `split_fused_moe` estimates local
  memory to decide expert fusion.
- Fully custom fused block (wqkv/w13, no standalone norm weights) →
  `FalconPerceptionConverter`.

## Gotchas

- `gen_config` **refuses to run under the original model dir** (removes the
  fresh out dirs and raises) — never point `-o` inside the source path.
- `bm1690` enables `fused_mlp` (unless gptq/awq non-4-bit); other chips don't.
- `num_core` defaults to the chip's max (`tpu_info.max_core_num`); `MASK_SIZE =
  npu_num * 4` sizes the static small mask used in dynamic mode.
- `--embedding_disk` skips embedding nets and writes `config/embedding.bin`
  (bf16/f16 per quantize) for CPU-side embedding in the demo.
- Demos feature-detect everything (`block_kv_0` exists, lm_head output dim,
  embedding input seq dim) — changing net names/orders on the compiler side
  silently breaks every demo's `chat.cpp`. The contract in
  `docs/LLM_Convert_Pipeline.md` is what must stay stable.
- `lora` implies logits lm_head: `lmhead_with_topk = not (do_sample or lora)`,
  so lora builds also get `greedy_head`/`sample_head`.
- `batch > 1` (`use_insert`) changes position_ids to `[1,1,len]`-style shapes
  and writes KV in place instead of returning it.

## Debugging entry points

- bmodel/demo mismatch → contract tables in `docs/LLM_Convert_Pipeline.md`,
  then demo `chat.cpp` net lookups vs `model_tool --info <bmodel>` (a
  `model.log` is auto-written next to the combined bmodel).
- "Unsupported model type" → source `config.json` `model_type` (or GGUF
  `general.architecture` + `ARCH_TO_MODEL_TYPE`) vs `LLM_CONVERTERS`.
- Accuracy problems → which branch of `init_quantization` ran (was quantize
  forced? what `q_group_size`?), then `support/debug*/` helpers in this repo.
- Compile failure → `<bmodel_dir>/<piece>.log`; rerun with `--again` to resume,
  `--debug` to serialize MLIR gen and keep intermediates.
