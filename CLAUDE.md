# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

LLM-TPU deploys LLMs and VLMs onto SOPHGO TPU chips (BM1684X / BM1688 / CV186X). The flow is:

```
HuggingFace/GGUF weights --tpu-mlir/llm_convert.py--> bmodel (quantized) --tpu-runtime--> PCIE/SoC inference
```

Compilation requires the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) toolchain (`llm_convert.py`), typically run inside the `sophgo/tpuc_dev` Docker image on an x86 host (no TPU is needed to compile). Pre-compiled bmodels can be downloaded instead of compiling. This repo itself contains **demo/eval code**, not the compiler.

Target chips are selected with `-c bm1684x | bm1688 | cv186x` (the converter also accepts `bm1690` / `bm1684x2` for TPU v7). Quantize types: `bf16`, `f16`, `w8bf16`, `w4bf16`, `w8f16`, `w4f16` — `-q` defaults to `auto`, which follows the source weights' quantization. Prefer AWQ/GPTQ-quantized source weights for best accuracy.

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
Finds every `CMakeLists.txt` under `models/` (skipping an explicit exclude list) and runs `cmake .. && make -j4` in each. Use this to verify a change doesn't break the build across models.

### Compile a model to bmodel (requires TPU-MLIR env)
```bash
llm_convert.py -m /path/to/weights -s 2048 --max_input_length 1024 -c bm1684x -o out_dir
```
Key flags: `-m/--model_path` (required; HF directory or `.gguf` file) and `-s/--seq_length` (required) are the only mandatory arguments. `-c/--chip` defaults to `bm1684x`, `-o/--out_dir` defaults to `./<model>_<chip>_<quantize>`, and `-q/--quantize` defaults to `auto` (follows the source weights' quantization — omit `-q` for AWQ/GPTQ sources). Others: `--dynamic`, `--num_device N` (multi-chip), `--use_history_kv` + `--chunk_length` (reuse history KV cache during prefill; forces `--dynamic` and derives `--max_input_length` from `--chunk_length`), `--do_sample`, `--max_pixels` (VLM).

### Eval VLM accuracy
```bash
pip3 install -r eval/requirements.txt
python3 eval/eval_qwen3vl.py --model_path <model> --datasets <dataset>
```
`harness/` contains dataset runners (C-Eval, MMLU, Hisence) driven by `harness/task/bmodel_task.py` + `harness/tools/indicators.py`; install with `pip3 install datasets jieba nltk rouge-score`.

## Repository layout

- `models/` — one directory per supported model (Qwen3, Qwen2_5_VL, InternVL3, MiniCPMV4_6, Gemma4, LocateAnything, ...). Each typically has:
  - `README.md`, `run_demo.sh`, `config/` (tokenizer/chat_template/generation_config used at runtime)
  - `python_demo/` — `pipeline.py` (orchestration + tokenization via `transformers`) + `chat.cpp` (pybind11 TPU runtime) + `CMakeLists.txt`
  - `cpp_demo/` (only some models) — standalone C++ demo with bundled `lib_pcie`/`lib_soc`/`include`; either a single `demo.cpp` (newer models) or `pipeline.cpp` + `chat.cpp`/`chat.hpp`; builds a `pipeline` binary linking `bmrt`/`bmlib` + bundled tokenizer libs
  - variant dirs: `python_demo_parallel` (multi-chip TP), `python_demo_share_prompt`, `cpp_demo_multiuser`, `python_demo_multiimage`, `cpp_demo_pp` (pipeline-parallel), `python_demo_v7` (TPU v7 runtime)
- `support/` — `include/` + `include_v7/` (bmrt headers, reference only — **do not include from demos**), `lib_pcie/` + `lib_soc/` (replacement `libbmrt.so`/`libbmlib.so` if the system lib is too old), `tools/` (upload/export_lora), `debug/` + `debug_v7/` (debugging helpers when results mismatch).
- `harness/` — accuracy benchmark harness over datasets; `task/bmodel_task.py` loads a model's `pipeline.py` `Model` + `chat` module and scores with `tools/indicators.py`.
- `eval/` — standalone VLM accuracy scripts (CUDA source model vs BM1684X bmodel).
- `docs/` — `Quick_Start.md`, `FAQ.md`, `LLM_Convert_Pipeline.md`.
- `run.sh` — top-level demo launcher (small fixed name→dir map).
- `regression/run.sh` — build-all-demos check.

## tpu-mlir converter relationship

The bmodels run by these demos are produced by `llm_convert.py` in the tpu-mlir repo (`python/tools/llm_convert.py` + `python/llm/*Converter.py`; checked out at `/workspace/tpu-mlir` in this environment — consult it directly to verify flags). The contract between compiler and demos — net names inside the bmodel (`embedding`, `block_<i>`, `block_cache_<i>`, `block_kv_<i>`, `lm_head`, `greedy_head`/`sample_head`, `vit`), their I/O shapes, the `config/` dir contents, and which compile flag enables which runtime feature — is documented in [docs/LLM_Convert_Pipeline.md](docs/LLM_Convert_Pipeline.md). Consult it before adding a model, changing demo net I/O, or debugging a bmodel/demo mismatch.

## Architecture notes

- **Python ↔ TPU split**: `pipeline.py` handles prompt construction, tokenization, and detokenization using `transformers` (`AutoTokenizer`/`AutoProcessor`) loaded from the model's `config/` dir. The actual TPU inference (bmodel load, prefill, decode, KV-cache, sampling) lives in `chat.cpp`, exposed to Python as a pybind11 module named `chat` (class e.g. `chat.Qwen2_5VL()`, `chat.Qwen()`). When adding/changing a model's runtime behavior, edit `chat.cpp`; when changing prompt/IO flow, edit `pipeline.py`.
- **Runtime libs**: demos link `bmrt` and `bmlib` (from `/opt/sophon/libsophon-current` on the device/docker). Headers in `support/include*` are for reference only. TPU v7 demos use `include_v7`/`debug_v7` and a separate runtime.
- **bmodel + config pairing**: a compiled bmodel is always run together with the original `config/` (tokenizer, chat template, generation config) — `pipeline.py` loads the tokenizer from `--config_path`, not from the bmodel.
- **Multi-chip / advanced features** are compile-time options on `llm_convert.py` (`--num_device`, `--use_history_kv`, `--dynamic`) and have matching demo variants under the model dir.

## Code style

- C++ is clang-format (LLVM base) and clang-tidy configured at the repo root (`.clang-format`, `.clang-tidy`); identifiers are `camelBack` for members/params/variables. Demo `CMakeLists.txt` build with `-Wall -Werror`, so warnings will fail the build.
- C++ standard is C++17 (`gnu++17`); `.vscode/c_cpp_properties.json` adds `support/include` + `support/include_v7` for IntelliSense.
- Python demos use `yapf: disable/enable` markers around literal message dicts — preserve those when editing `pipeline.py` message construction.

## Working style

- **English refinement:** Users are mostly non-native English speakers. When the user's input or a description contains awkward or incorrect English, render the corresponding output (reports, docs, commit messages) in clear, natural English rather than mirroring the broken phrasing. If the user's English is already correct, preserve it as-is.
- **No compiling:** Do not try to compile this project (no `cmake`/`make`, `regression/run.sh`, or syntax-check builds) — this environment has no SOPHGO toolchain or TPU hardware. Verify C++ changes by reviewing and diffing against reference code (e.g. a model's `cpp_demo`) instead.
- **No auto-commit:** When making code fixes, do not `git commit` them directly. Leave the changes in the working tree for the user to review and commit themselves.
- **Preserve file ownership:** Do not change file ownership. Edits made through the Edit/Write tools run as root and silently change the edited file's owner to `root` — after editing, copying, moving, or regenerating any file, restore its original owner (repo files are uid/gid 1018; verify against untouched neighbors with `ls -l`), e.g. `chown 1018:1018 <files>`.
- **Remember in CLAUDE.md:** When the user asks to remember something (a rule, preference, or lesson learned), always record it in this `CLAUDE.md` so it persists in the repo for every session — not in private/session-only memory.
