# MiniCPM5

This project implements the deployment of the large model [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) on BM1684X/CV84X6. The model is converted into a bmodel via the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed to a PCIE environment or a SoC environment using C++ code.

MiniCPM5-2B is a standard `LlamaForCausalLM` (2.5B params, 42 layers, hidden 2048, GQA with 16 query heads / 2 KV heads, 131072 native context, rope_theta 5e6, ChatML prompt format). Because its `config.json` declares `model_type: "llama"` with pure-Llama weight names, `llm_convert.py` already supports it out of the box — no compiler-side adaptation is required.

## Compile the LLM Model

This section describes how to compile an LLM into a bmodel.

#### 1. Download MiniCPM5-2B from HuggingFace

(Relatively large; will take a long time)

```shell
git lfs install
git clone git@hf.co:openbmb/MiniCPM5-2B
```

Note: the safetensors must be fully downloaded (`git lfs pull`); a 135-byte
LFS pointer stub will fail with `Error while deserializing header`.

#### 2. Download docker and start the container

```shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

The following assumes that the environment is in the `/workspace` directory of the docker container.

#### 3. Download the `TPU-MLIR` code and compile it

(You can also directly download and extract the pre-compiled release package)

```shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  # activate the environment variables
./build.sh # compile mlir
```

#### 4. Compile the model to generate the bmodel

```shell
llm_convert.py -m /workspace/MiniCPM5-2B -s 512 --quantize bf16 -c bm1684x --out_dir minicpm5_2b
```

Important notes on quantization:

- The official MiniCPM5-2B release ships BF16 weights with **no**
  `quantization_config` in its config, so `-q auto` (the default) fails with
  "No quantization config found". Pass an explicit quantize flag:
  - `--quantize bf16` — recommended, matches the released dtype.
  - `--quantize w4bf16` / `w8bf16` — compiler-side PTQ; works but accuracy is
    not guaranteed on a non-prequantized checkpoint (the tool prints the same
    warning).
  - `--quantize f16` fails the dtype-mismatch check (weights are bfloat16).

For longer sequences, add `--dynamic` so the first-token latency scales with
the actual input length:

```shell
llm_convert.py -m /workspace/MiniCPM5-2B -s 8192 --quantize bf16 -c bm1684x --dynamic --out_dir minicpm5_2b
```

For multi-turn conversations or long contexts, prefer the history-KV mode
(compiles three instruction groups: `block_` prefill, `block_kv_` prefill with
history, `block_cache_` decode):

```shell
llm_convert.py -m /workspace/MiniCPM5-2B -s 8192 --quantize bf16 -c bm1684x --use_history_kv --chunk_length 1024 --out_dir minicpm5_2b
```

After compilation, `minicpm5-xxx.bmodel` and `config` are generated in the
specified directory `minicpm5_2b`. Keep `out_dir` outside the model path —
the config copy step refuses to run inside the original model directory.

## Compile and Run the Program

Please copy the program to the PCIE environment or SoC environment before compiling. Then copy `minicpm5-xxx.bmodel` and `config` over.

#### python demo

Compile the library files to generate the `chat.cpython*.so` file, and copy this file to the directory containing `pipeline.py`

```shell
cd python_demo
mkdir build
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

* python demo

```shell
python3 pipeline.py -m minicpm5_xxx.bmodel -c config
```

`model` is the actual model storage path; `config` is the configuration file
path (the directory copied from the compile step).

* one-shot / scripted mode

```shell
python3 pipeline.py -m minicpm5_xxx.bmodel -c config -p "你好，请简单介绍一下你自己。"
```

* sampling mode (greedy by default)

```shell
python3 pipeline.py -m minicpm5_xxx.bmodel -c config --do_sample
```

MiniCPM5's `generation_config.json` has no `top_k`, so the demo falls back to
`top_k = 5` in sampling mode; `temperature` (1.0) and `top_p` (0.95) come from
the config.

## Notes on the MiniCPM5 chat format

- The ChatML template lives in a standalone `chat_template.jinja` next to
  `tokenizer_config.json`; both are copied into `out_dir/config` by the
  compile step. `apply_chat_template` needs a reasonably new `transformers`
  (the `chat_template.jinja` auto-load needs >= 4.43; the tpuc_dev container
  ships a new enough version).
- The template already emits `<s>` itself while the tokenizer also adds a bos
  by default, so `pipeline.py` encodes with `add_special_tokens=False` to
  avoid a duplicated bos token.
- MiniCPM5 ends assistant turns with `<|im_end|>` (id 130073), not the base
  eos `</s>` (id 1). `pipeline.py` stops on both ids in every mode, otherwise
  greedy decoding would never stop.
