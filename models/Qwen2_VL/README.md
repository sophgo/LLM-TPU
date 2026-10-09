# Qwen2-VL

This project deploys the multimodal large model [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct-AWQ) on BM1684X/BM1688. The model is converted to bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed to the PCIE environment or SoC environment of BM1684X using C++ code.

## Development Environment Setup

#### 1. Download `Qwen2-VL-2B-Instruct-AWQ` from HuggingFace

(Relatively large, will take a long time)

``` shell
git lfs install
git clone git@hf.co:Qwen/Qwen2-VL-2B-Instruct-AWQ
```

## Compile the Model

This section describes how to compile the onnx model into bmodel. You can also skip the compilation step and directly download the compiled model:

``` shell
# 1684x, 672x896
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2-vl-2b-instruct-awq_w4bf16_seq2048_bm1684x_1dev_20250616_180313.bmodel

# 1688, 672x896
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2-vl-2b-instruct-awq_w4bf16_seq2048_bm1688_2core_20250616_193802.bmodel
```

#### 1. Download docker and start the container

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash

docker exec -it myname1234 bash
```
The following assumes that all environments are in the `/workspace` directory of docker.

#### 2. Download the `TPU-MLIR` code and compile it

(You can also directly download and extract the precompiled release package)

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  # activate environment variables
./build.sh # compile mlir
```

#### 3. Compile the model to generate bmodel

``` shell
# If prompted about transformers version issues, run pip3 install transformers -U
# bm1688 requires specifying -c bm1688; -g specifies the group size; AWQ and GPTQ actually do not require specifying group_size; non-AWQ/GPTQ models do
llm_convert.py -m /workspace/Qwen2-VL-2B-Instruct-AWQ -s 2048 --quantize w4f16 -c bm1684x --out_dir qwen2vl_2b --max_pixels 672,896
```
After compilation, `qwen2vl_2b-xxx.bmodel` and `config` are generated in the specified directory `qwen2vl_2b`

## Build and Run the Program

Compile the library files to generate the `chat.cpython*.so` file, and copy it to the directory of the `pipeline.py` file;
also copy the bmodel file and config directory there

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

* python demo

``` shell
python3 pipeline.py -m qwen2-vl-2b-xxxx.bmodel -c config
```

---

## Compatible Models

The `python_demo/` here is **architecture-level reusable** for any `qwen2_vl`-family model whose bmodel is compiled by `llm_convert.py` with the standard 3-input ViT interface (input_states / position_ids / full_attn_mask). It is not specific to the original Qwen2-VL weights — only the bmodel + `config/` directory change.

### MinerU2.5-Pro (document parsing)

[MinerU2.5-Pro](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B) is a 1.2B document-parsing VLM built on the Qwen2-VL architecture (24 layers, hidden 896, GQA 14/2, ViT depth 32 hidden 1280). It is **fully compatible** with this demo; no source change is required.

#### Compile

First download the source weights (ModelScope mirror is faster than HuggingFace in mainland China; both work):

``` shell
git clone https://www.modelscope.cn/OpenDataLab/MinerU2.5-Pro-2605-1.2B.git
# or: git clone https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B
```

Set up the TPU-MLIR environment as described in [Compile the Model](#compile-the-model) above (steps 1 & 2: pull the `sophgo/tpuc_dev` Docker image, clone tpu-mlir, `source ./envsetup.sh`). Then run `llm_convert.py`:

``` shell
llm_convert.py -m /path/to/MinerU2.5-Pro-2605-1.2B \
  -s 4096 --max_input_length 4096 \
  -c bm1684x -q bf16 \
  --max_pixels 1260,1260 \
  -o ./out
```

This produces `mineru2.5-pro-2605-1.2b_bf16_seq4096_bm1684x_1dev_static_*.bmodel` and a `config/` directory.

#### Download pre-compiled bmodel

If you don't want to compile, pre-built bmodels (bf16, seq=4096, max_pixels=1260×1260) are published on the SOPHGO file server:

```shell
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

# BM1684X (PCIe & SoC share the same bmodel)
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/mineru2.5-pro-2605-1.2b_bf16_seq4096_bm1684x_1dev_static_20261008_161704.bmodel

# BM1688 (2core)
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/mineru2.5-pro-2605-1.2b_bf16_seq4096_bm1688_2core_static_20261008_173415.bmodel

# CV84X6 (4core)
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/mineru2.5-pro-2605-1.2b_bf16_seq4096_cv84x6_4core_static_20261008_181859.bmodel
```

You will also need the tokenizer/processor configuration directory. Reuse this demo's `config/` if you have it; otherwise extract it from the original model (no safetensors needed) — `config.json`, `tokenizer.json`, `tokenizer_config.json`, `preprocessor_config.json`, `chat_template.jinja`, `added_tokens.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt`.

#### Run

``` shell
cd python_demo
python3 pipeline.py -m /path/to/mineru2.5-pro-2605-1.2b_*.bmodel \
                    -c /path/to/out/config \
                    -d 0
```

#### Environment requirements

Install the Python packages the demo imports (plus `pybind11` for the build):

```shell
pip3 install --user --no-cache-dir torch torchvision transformers qwen_vl_utils pybind11
```

Two version constraints to watch for:

- **`transformers>=4.56`** (the model card's requirement; `transformers 5.x` also works).
- **`jinja2>=3.1`** — if `transformers>=5.0` is installed and the system jinja2 is older than 3.1, `apply_chat_template` fails with `ImportError: apply_chat_template requires jinja2>=3.1.0`. Fix: `pip3 install --user --no-cache-dir 'jinja2>=3.1'`.
- On aarch64 SoC boards, **`scikit-learn` must NOT be installed** — its 1.7.x wheel has a static-TLS bug that surfaces when `transformers` imports it. If you see `libgomp...so: cannot allocate memory in static TLS block` from sklearn, run `pip3 uninstall scikit-learn -y`. `transformers` only uses sklearn for niche eval metrics and works fine without it.

#### Prompt format (critical)

MinerU is **task-prompt-driven**: the same weights behave as a generic captioner vs. a structured OCR engine depending on the user-prompt prefix. The defaults baked into the demo chat template produce only a brief description; for document parsing you must use the official task prefixes below.

| Task | User prompt prefix (literally prepend to your text) |
|---|---|
| Plain-text OCR (default) | `\nText Recognition:` |
| Layout detection | `\nLayout Detection:` |
| Table recognition | `\nTable Recognition:` |
| Formula recognition | `\nFormula Recognition:` |
| Image / chart analysis | `\nImage Analysis:` |

The system prompt is the standard `You are a helpful assistant.` — no change needed.

Reference: these strings come from the official [`mineru-vl-utils`](https://github.com/opendatalab/mineru-vl-utils) `DEFAULT_PROMPTS` table. MinerU's official pipeline (`two_step_extract`) is **two passes** over the image — one `Layout Detection` pass to get block bboxes, then a per-block `Text / Table / Formula / Image Recognition` pass — so for production use you would loop the demo's image+prompt path with cropped regions. For a single-page quick test, the `\nText Recognition:` prompt alone already produces full-page OCR.

#### Example (single-page OCR)

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image", "image": "page.png"},
        {"type": "text",  "text": "\nText Recognition:"},
    ]},
]
```

The model emits a flat text stream using `<fcel>` (first-cell), `<lcel>` (line-cell), `<nl>` (new-line) etc. as inline structural markers; `mineru-vl-utils` post-processes them into Markdown. The demo prints the raw stream — downstream formatting is left to the caller.
