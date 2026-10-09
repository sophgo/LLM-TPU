# bge-reranker-v2-m3

This project deploys [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) on BM1684X / BM1688 / CV84X6. The model is compiled to a bmodel with the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) toolchain and run with a Python demo on PCIE or SoC.

bge-reranker-v2-m3 is a **non-generative reranker model** (XLM-RoBERTa-large based, ~568M parameters). It takes a query-document pair and outputs a relevance score (logit) that can be mapped to [0, 1] via sigmoid. The model supports 100+ languages and is designed for re-ranking candidate documents in retrieval pipelines.

Architecture highlights compared to decoder-only LLMs:

- **Encoder-only** (XLMRobertaForSequenceClassification): 24 transformer layers, hidden size 1024, 16 attention heads
- **Bidirectional attention** (no causal mask) with additive padding mask
- **Post-LN** (LayerNorm after residual, not before like RMSNorm in decoder LLMs)
- **Absolute position embeddings** (learned, not RoPE); `padding_idx=1`
- **Classifier head**: CLS token → dense(1024→1024) → tanh → dense(1024→1) → logit

Supported chips: BM1684X (PCIe + SoC), BM1688 (SoC), and CV84X6 (SoC). BM1684X PCIe/SoC share one bmodel; BM1688 and CV84X6 each require a separately compiled bmodel.

## Download pre-compiled bmodels

If you don't want to compile from source, download pre-compiled bmodels via `dfss`:

```shell
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade --quiet

# BM1684X (PCIe + SoC shared)
python3 -m dfss --url=open@sophgo.com:ext_model_information/LLM/LLM-TPU/bge-reranker-v2-m3_bf16_seq2048_bm1684x_1dev_static_20261008_114530.bmodel

# BM1688
python3 -m dfss --url=open@sophgo.com:ext_model_information/LLM/LLM-TPU/bge-reranker-v2-m3_bf16_seq2048_bm1688_2core_static_20261008_144830.bmodel

# CV84X6
python3 -m dfss --url=open@sophgo.com:ext_model_information/LLM/LLM-TPU/bge-reranker-v2-m3_bf16_seq2048_cv84x6_4core_static_20261009_110923.bmodel
```

## Compile the bmodel

#### 1. Download the model weights

```shell
# HuggingFace
git lfs install
git clone https://huggingface.co/BAAI/bge-reranker-v2-m3
# Or ModelScope
pip3 install modelscope
modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir ./bge-reranker-v2-m3
```

#### 2. Start the TPU-MLIR docker

```shell
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

The following assumes work is done in `/workspace` inside the container.

#### 3. Download the `TPU-MLIR` code and build it

```shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir && source ./envsetup.sh && ./build.sh
```

#### 4. Compile the model to generate the bmodel

```shell
# BM1684X, BF16, seq=2048
llm_convert.py -m /workspace/bge-reranker-v2-m3 \
  -s 2048 -c bm1684x -q bf16 --out_dir bge_reranker_v2_m3

# BM1688, BF16, seq=2048
llm_convert.py -m /workspace/bge-reranker-v2-m3 \
  -s 2048 -c bm1688 -q bf16 --out_dir bge_reranker_v2_m3_1688

# CV84X6, BF16, seq=2048
llm_convert.py -m /workspace/bge-reranker-v2-m3 \
  -s 2048 -c cv84x6 -q bf16 --out_dir bge_reranker_v2_m3_cv84x6
```

> **Note**: this model requires `XLMRobertaConverter` in tpu-mlir (encoder-only architecture). If your tpu-mlir version does not include it, cherry-pick or add `python/llm/XLMRobertaConverter.py` before compiling.

After compilation, `bge-reranker-v2-m3_*.bmodel` and `config/` are generated in the output directory.

## Build and run the Python demo

Copy the bmodel and `config/` to the PCIE / SoC machine, then:

```shell
cd python_demo
mkdir build && cd build && cmake .. && make -j4 && cp *cpython* .. && cd ..
```

### Reranking demo

```shell
python3 pipeline.py -m bge-reranker-v2-m3_*.bmodel -c ../config --normalize \
  --query "what is panda?" \
  --documents "hi" \
              "The giant panda is a bear species endemic to China."
```

Example output:

```
Results (normalized):
  [0.8459] The giant panda is a bear species endemic to China.
  [0.0003] hi
```

### Interactive mode

```shell
python3 pipeline.py -m bge-reranker-v2-m3_*.bmodel -c ../config
```

Commands inside the interactive loop:

| Command | Action |
| :--- | :--- |
| `<query> \|\|\| <document>` | Compute relevance score (raw logit) |
| `/norm` | Toggle sigmoid normalization on/off |
| `/q` · `/exit` | Quit |

## CLI parameters

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `-m, --model_path` | — | Path to the bmodel file (required) |
| `-c, --config_path` | `../config` | Tokenizer config directory |
| `-d, --devid` | `0` | TPU device id |
| `--normalize` | off | Apply sigmoid to map logit to [0, 1] |
| `--query` | — | Query text for reranking demo |
| `--documents` | — | Document texts for reranking demo (space-separated) |

## Notes

- **Tokenization**: the tokenizer uses pair format `[[query, document]]` which produces `<s> query </s></s> document </s>` (XLM-RoBERTa SentencePiece). Maximum input length is 2048 tokens (compile-time `seq_length`).
- **Device memory**: ~1117 MB on BM1684X, ~1164 MB on BM1688, ~1132 MB on CV84X6 (bf16, static, seq=2048).
- **Latency**: ~550 ms/pair on BM1684X PCIe, ~1110 ms on BM1684X SoC, ~2200 ms on BM1688 SoC, ~860 ms on CV84X6 SoC (static compilation; all 2048 positions are computed regardless of actual token count).
- **Precision**: bf16 preserves reranker accuracy well — logit difference vs. HuggingFace reference is typically <0.1, sigmoid scores match within ±0.01. BM1684X PCIe/SoC produce identical scores; BM1688 and CV84X6 differ by at most ±0.003 in sigmoid space.

> **CV84X6 known issue**: tpu-mlir renamed the chip identifier from `bm1684x2` to `cv84x6` (2026-09-22), but some board runtimes still check for the old name. If `bmrt_load_bmodel` fails on CV84X6, apply this workaround in `lib/Dialect/Tpu/Transforms/Codegen/BM168xCodegen.cpp` and rebuild `tpuc-opt`/`tpuc-tool`:
> ```cpp
> chip = module::stringifyChip(chip_).upper();
> if (chip == "CV84X6") chip = "BM1684X2";
> ```
> Then recompile the bmodel. This is a temporary fix until the board runtime is updated.
