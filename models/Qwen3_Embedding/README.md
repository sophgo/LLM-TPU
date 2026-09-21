# Qwen3-Embedding-0.6B

This project deploys [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) on BM1684X / BM1688 / CV84X6. The model is compiled to a bmodel with the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) toolchain and run with a Python demo on PCIE or SoC.

Qwen3-Embedding is a **non-generative text embedding model**. It takes a text (a query or a document) and outputs a 1024-dim vector via last-token pooling + L2 normalization. Use cases include semantic search, retrieval, and clustering. It is instruction-aware (queries get an instruction prefix, documents do not) and supports MRL (the output can be truncated to any dimension in 32–1024).

Supported chips: BM1684X (PCIe + SoC), BM1688 (SoC), and CV84X6 (SoC). BM1684X PCIe/SoC share one bmodel; BM1688 and CV84X6 each require a separately compiled bmodel.

## Download pre-compiled bmodel

```shell
# BM1684X
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-embedding-0.6b_bf16_seq8192_bm1684x_1dev_static_20260916_174942.bmodel

# BM1688
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-embedding-0.6b_bf16_seq8192_bm1688_2core_static_20260917_101539.bmodel

# CV84X6
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-embedding-0.6b_bf16_seq8192_bm1684x2_4core_static_20260920_184312.bmodel
```

## Compile the bmodel

#### 1. Download the model weights

```shell
# HuggingFace
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
# Or ModelScope
pip3 install modelscope
modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir ./Qwen3-Embedding-0.6B
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
# BM1684X, BF16, seq=8192
llm_convert.py -m /workspace/Qwen3-Embedding-0.6B \
  -s 8192 -c bm1684x -q bf16 --out_dir qwen3_embedding_0.6b

# BM1688, BF16, seq=8192
llm_convert.py -m /workspace/Qwen3-Embedding-0.6B \
  -s 8192 -c bm1688 -q bf16 --out_dir qwen3_embedding_0.6b_1688

# CV84X6, BF16, seq=8192
llm_convert.py -m /workspace/Qwen3-Embedding-0.6B \
  -s 8192 -c bm1684x2 -q bf16 --out_dir qwen3_embedding_0.6b_cv84x6
```

For batch encoding of variable-length texts, compile with `--dynamic` so the per-encode latency scales with the actual input length instead of always running 8192:

```shell
llm_convert.py -m /workspace/Qwen3-Embedding-0.6B \
  -s 8192 -c bm1684x -q bf16 --dynamic --out_dir qwen3_embedding_0.6b_dyn
```

After compilation, `qwen3-embedding-*.bmodel` and `config/` are generated in the output directory.

## Build and run the Python demo

Copy the bmodel and `config/` to the PCIE / SoC machine, then:

```shell
cd python_demo
mkdir build && cd build && cmake .. && make -j4 && cp *cpython* .. && cd ..
```

### Retrieval demo

```shell
python3 pipeline.py -m qwen3-embedding-xxx.bmodel -c ../config \
  --query "What is deep learning?" \
  --documents "Deep learning is a subset of machine learning" \
              "The weather is nice today" \
              "Neural networks have many layers"
```

Example output:

```
Similarity (dim=1024):
  [0.6386] Deep learning is a subset of machine learning
  [0.5259] Neural networks have many layers
  [0.1491] The weather is nice today
```

### Interactive mode

```shell
python3 pipeline.py -m qwen3-embedding-xxx.bmodel -c ../config
```

Commands inside the interactive loop:

| Command | Action |
| :--- | :--- |
| `<text>` | Encode text (with default instruction), print embedding + norm |
| `/sim <t1> \|\|\| <t2>` | Cosine similarity between two texts |
| `/dim <N>` | Set MRL output dimension (32–1024) |
| `/nodim` | Reset dim to full hidden size (1024) |
| `/q` · `/exit` | Quit |

## CLI parameters

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `-m, --model_path` | — | Path to the bmodel file (required) |
| `-c, --config_path` | `../config` | Tokenizer config directory |
| `-d, --devid` | `0` | TPU device id |
| `--dim` | `1024` | Output embedding dimension (MRL truncation, 32–1024) |
| `--query` | — | Query text for retrieval demo |
| `--documents` | — | Document texts for retrieval demo (space-separated) |
| `--instruction` | built-in web-search | Custom instruction prefix for queries |

## Notes

- **Instruction prefix**: by default queries are prefixed with `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:` (the model's recommended web-search instruction); documents are encoded without a prefix. Override with `--instruction`.
- **MRL truncation**: `--dim N` truncates the 1024-dim output to the first `N` dims and re-normalizes. The bmodel is unaffected — truncation is done on the host. `N` should be in 32–1024; values outside this range trigger a warning.
- **Latency**: static seq=8192 bmodels compute all 8192 positions per encode, so latency is fixed regardless of the actual token count. Use `--dynamic` for production batch encoding.
