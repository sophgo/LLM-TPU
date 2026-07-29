# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

LLM-TPU deploys LLMs and VLMs onto SOPHGO TPU chips (BM1684X / BM1688 / CV186X). The flow is:

```
HuggingFace/GGUF weights --tpu-mlir/llm_convert.py--> bmodel (quantized) --tpu-runtime--> PCIE/SoC inference
```

This repo contains **demo/eval code**, not the compiler. Compilation uses the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) toolchain (`llm_convert.py`) inside the `sophgo/tpuc_dev` Docker image on an x86 host (no TPU needed to compile); pre-compiled bmodels can be downloaded instead.

## Common commands

### Run a built-in demo end-to-end
```bash
./run.sh --model qwen3        # also: qwen2.5vl, internvl3
```
`run.sh` maps the short name to `models/<Dir>/run_demo.sh`, which downloads a pre-compiled bmodel via `dfss`, builds the python extension, and launches `python_demo/pipeline.py`.

### Build a single model's python demo
Each demo builds a pybind11 module (`chat.cpp` -> `chat*.cpython*.so`) consumed by `pipeline.py`:
```bash
cd models/<Model>/python_demo
rm -rf build && mkdir build && cd build && cmake .. && make
cp *cpython* ..
python3 pipeline.py --model_path <x.bmodel> --config_path ./config --devid 0
```
CMake links against `bmrt` / `bmlib` from `/opt/sophon/libsophon-current` and requires pybind11. `TARGET_ARCH` defaults to `pcie` (also `soc`).

### Build all demos (regression compile check)
```bash
./regression/run.sh
```
Finds every `CMakeLists.txt` under `models/` (skipping an exclude list) and runs `cmake .. && make -j4` in each.

### Compile a model to bmodel (requires TPU-MLIR env)
Before running `llm_convert.py` or any tpu-mlir tool, source the environment:
```bash
cd /workspace/llm/tpu-mlir/ && source ./envsetup.sh
```
Then:
```bash
llm_convert.py -m /path/to/weights -s 2048 --max_input_length 1024 -c bm1684x -o out_dir
```
Only `-m/--model_path` and `-s/--seq_length` are mandatory; `-c` defaults to `bm1684x`, `-q` to `auto` (follows the source weights' quantization — omit `-q` for AWQ/GPTQ sources). For all other flags and how each maps to a runtime feature, use the `tpu-mlir-llm-converter` skill.

### Eval accuracy
```bash
pip3 install -r eval/requirements.txt
python3 eval/eval_qwen3vl.py --model_path <model> --datasets <dataset>   # VLM
```
`harness/` scores LLMs over datasets (C-Eval, MMLU, Hisence) via `task/bmodel_task.py` + `tools/indicators.py`; install with `pip3 install datasets jieba nltk rouge-score`.

## Repository layout

- `models/` — one directory per supported model (Qwen3, Qwen2_5_VL, InternVL3, MiniCPMV4_6, Gemma4, LocateAnything, ...). Each typically has:
  - `config/` — runtime assets loaded by `pipeline.py` via `transformers`: `config.json` (architecture), `tokenizer.json` / `tokenizer_config.json` / `vocab.json` (tokenization), `generation_config.json` (sampling params), and the chat template. These files are **not** embedded in the bmodel — they must ship alongside it.
  - `python_demo/` — `pipeline.py` (orchestration + tokenization via `transformers`) + `chat.cpp` (pybind11 TPU runtime) + `CMakeLists.txt`
  - `cpp_demo/` (only some models) — standalone C++ demo with bundled `lib_pcie`/`lib_soc`/`include`; either a single `demo.cpp` (newer models) or `pipeline.cpp` + `chat.cpp`/`chat.hpp`; builds a `pipeline` binary linking `bmrt`/`bmlib` + bundled tokenizer libs
  - variant dirs — each enabled by a specific `llm_convert.py` compile flag:

    | Directory suffix | Compile flag(s) | Purpose |
    |---|---|---|
    | `python_demo_parallel` / `cpp_demo_parallel` | `--num_device N` | Multi-chip tensor parallelism |
    | `python_demo_share_prompt` | `--use_history_kv --chunk_length N` | Reuse history KV cache across turns (forces `--dynamic`) |
    | `cpp_demo_multiuser` | (runtime only) | Load same model N times, weights shared on chip |
    | `python_demo_multiimage` | (runtime only) | Batch image processing for VLMs |
    | `cpp_demo_pp` | `--distribute_strategy pp` | Pipeline parallelism (per-layer distribution) |
    | `python_demo_v7` | (separate TPU v7 toolchain) | TPU v7 runtime with `tpuv7_*.h` headers |
- `support/` — `include/` + `include_v7/` (bmrt headers, reference only — **do not include from demos**), `lib_pcie/` + `lib_soc/` (replacement `libbmrt.so`/`libbmlib.so` if the system lib is too old), `tools/` (upload/export_lora), `debug/` + `debug_v7/` (debugging helpers when bmodel output mismatches a reference — copy `cnpy.cpp`/`cnpy.h` into a demo, link `libz`, call `dump_net_to_file` to export net I/O as `.npz` for offline comparison).
- `harness/` — accuracy benchmark harness over datasets; `task/bmodel_task.py` loads a model's `pipeline.py` `Model` + `chat` module and scores with `tools/indicators.py`.
- `eval/` — standalone VLM accuracy scripts (CUDA source model vs BM1684X bmodel).
- `docs/` — `FAQ.md`, `LLM_Convert_Pipeline.md`.
- `run.sh` / `regression/run.sh` — demo launcher / build-all-demos check.

## tpu-mlir converter relationship

The bmodels these demos run are produced by `llm_convert.py` in the tpu-mlir repo (checked out at `/workspace/tpu-mlir`). Three references, by question type:

- **Compiler internals** (dispatch, flag semantics, quant detection, gotchas): the `tpu-mlir-llm-converter` skill.
- **bmodel ↔ demo contract** (net names, I/O shapes, `config/` contents, which compile flag enables which runtime feature): [docs/LLM_Convert_Pipeline.md](docs/LLM_Convert_Pipeline.md). Consult it before adding a model, changing demo net I/O, or debugging a bmodel/demo mismatch.
- **Source of truth** for any flag: `/workspace/tpu-mlir/python/tools/llm_convert.py`.

## Adding a new model

The typical flow for porting a new LLM/VLM:

1. **Check the compiler supports it**: look up `model_type` (from the HF `config.json`) in the dispatch table in [docs/LLM_Convert_Pipeline.md](docs/LLM_Convert_Pipeline.md) — or check `/workspace/tpu-mlir/python/llm/*Converter.py` for existing converter classes. If no converter exists, one must be added to tpu-mlir first.
2. **Compile a bmodel**: `llm_convert.py -m /path/to/weights -s <seq_len> -c <chip> -q <quantize>`. The converter emits the bmodel + `config/` dir.
3. **Write the demo**: copy the closest existing model's `python_demo/` (and optionally `cpp_demo/`) and adapt:
   - `chat.cpp` — adjust net names, I/O shapes, and any model-specific logic (e.g. ViT encoding for VLMs, rotary embedding differences) to match what the converter emitted.
   - `pipeline.py` — adjust prompt construction, tokenization, and (for VLMs) image/video preprocessing to match the model's expected input format.
   - `CMakeLists.txt` — usually just change the project name.
4. **Verify** by diffing against a known-good reference (e.g. the model's `cpp_demo/` or a previously ported model with the same architecture).

The `/llm-porting` skill can bootstrap a new model template.

## Architecture notes

- **Python ↔ TPU split**: `pipeline.py` handles prompt construction, tokenization, and detokenization via `transformers` (`AutoTokenizer`/`AutoProcessor`) loaded from the model's `config/` dir. TPU inference (bmodel load, prefill, decode, KV-cache, sampling) lives in `chat.cpp`, exposed as a pybind11 module named `chat` (e.g. `chat.Qwen()`). Edit `chat.cpp` for runtime behavior, `pipeline.py` for prompt/IO flow.
- **Runtime libs**: demos link `bmrt` and `bmlib` from `/opt/sophon/libsophon-current`. TPU v7 demos use `include_v7`/`debug_v7` and a separate runtime.
- **bmodel + config pairing**: a compiled bmodel always runs with the original `config/` dir — never mix a bmodel with a config dir from a different model or tokenizer revision.
- **Multi-chip / advanced features** are compile-time options on `llm_convert.py` with matching demo variants under the model dir.

## Code style

- C++ is clang-format (LLVM base) and clang-tidy configured at the repo root (`.clang-format`, `.clang-tidy`); identifiers are `camelBack` for members/params/variables. Demo `CMakeLists.txt` build with `-Wall -Werror`, so warnings will fail the build.
- C++ standard is C++17 (`gnu++17`); `.vscode/c_cpp_properties.json` adds `support/include` + `support/include_v7` for IntelliSense.
- Python demos use `yapf: disable/enable` markers around literal message dicts — preserve those when editing `pipeline.py` message construction.

## Working style

- **English refinement:** Users are mostly non-native English speakers. When the user's input or a description contains awkward or incorrect English, render the corresponding output (reports, docs, commit messages) in clear, natural English rather than mirroring the broken phrasing. If the user's English is already correct, preserve it as-is.
- **No compiling:** Do not try to compile this project (no `cmake`/`make`, `regression/run.sh`, or syntax-check builds) — this environment has no SOPHGO toolchain or TPU hardware. Verify C++ changes by reviewing and diffing against reference code (e.g. a model's `cpp_demo`) instead.
- **No auto-commit:** When making code fixes, do not `git commit` them directly. Leave the changes in the working tree for the user to review and commit themselves.
- **Preserve file ownership:** Do not change file ownership. Edits made through the Edit/Write tools run as root and silently change the edited file's owner to `root` — after editing, copying, moving, or regenerating any file, restore its original owner (repo files are uid/gid 1001; verify against untouched neighbors with `ls -l`), e.g. `chown 1001:1001 <files>`.
- **Remember in CLAUDE.md:** When the user asks to remember something (a rule, preference, or lesson learned), always record it in this `CLAUDE.md` so it persists in the repo for every session — not in private/session-only memory.
