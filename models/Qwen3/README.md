# Qwen3

This project deploys the large model [Qwen3](https://huggingface.co/Qwen/Qwen3-4B-AWQ) on BM1684X/BM1688. The model is converted to a bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler and deployed with C++ code to a PCIE or SoC environment.


This document covers how to compile the bmodel and how to run it in a BM1684X/BM1688 environment. The LLM compilation step can be skipped by downloading directly from the following links:

``` shell
# 1684x 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-4b-awq_w4bf16_seq512_bm1684x_1dev_20250514_161445.bmodel
# 1684x 8k, static model
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-4b-awq_w4bf16_seq8192_bm1684x_1dev_20250514_161732.bmodel
# 1684x 8k, dynamic model
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-4b-awq_w4bf16_seq8192_bm1684x_dyn.bmodel
# 1684x 8k, supports prefill_with_kv, history saved as kv
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-4b-awq_w4bf16_seq8192_bm1684x_1dev_kv.bmodel

# 1688 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU_Lite/qwen3-4b-awq_w4bf16_seq512_bm1688_2core_20250514_162231.bmodel
```

### deepseek-r1-0528-qwen3-8b

``` shell
# 1684x 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-0528-qwen3-8b_w4bf16_seq512_bm1684x.tar
# 1684x 4k
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-0528-qwen3-8b_w4bf16_seq4096_bm1684x.tar
```

## Compile the LLM model

This section describes how to compile an LLM into a bmodel.

#### 1. Download `Qwen3-4B-AWQ` from HuggingFace

(Quite large; will take a long time)

``` shell
# Download the model
git lfs install
git clone git@hf.co:Qwen/Qwen3-4B-AWQ
# For 8B, use:
git clone git@hf.co:Qwen/Qwen3-8B-AWQ
```

#### 2. Download docker and start the container

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
The following assumes all work is done in the `/workspace` directory inside docker.

#### 2. Download the `TPU-MLIR` code and build it

(You can also directly download the pre-built release package and extract it)

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  # activate the environment variables
./build.sh # build mlir
```

#### 3. Compile the model to generate the bmodel

``` shell
# If prompted about a transformers version issue, run: pip3 install transformers -U
llm_convert.py -m /workspace/Qwen3-4B-AWQ -s 512 --quantize w4f16 -c bm1684x --out_dir qwen3_4b
```
After compilation, `qwen3-xxx.bmodel` and `config` are generated in the specified directory `qwen3_4b`.

In addition, if the specified seqlen is relatively long, e.g. 8K, you can compile with `--dynamic`; the first-token latency will then vary with the actual input length, as follows:
``` shell
# If prompted about a transformers version issue, run: pip3 install transformers -U
llm_convert.py -m /workspace/Qwen3-4B-AWQ -s 8192 --quantize w4f16 -c bm1684x --dynamic --out_dir qwen3_4b
```

History can be saved with KV Cache (the original approach uses tokens for history) by specifying `--use_history_kv`; also specify `--chunk_length`, e.g. 512.
When chunk_length is not long, the first-token latency of every conversation turn stays low, unaffected by the history length.
As follows:
``` shell
# If prompted about a transformers version issue, run: pip3 install transformers -U
llm_convert.py -m /workspace/Qwen3-4B-AWQ -s 8192 --quantize w4f16 -c bm1684x --use_history_kv --chunk_length 512 --out_dir qwen3_4b_kv
```

## Build and run the program

Please copy the program to a PCIE or SoC environment before compiling. Then copy `qwen3-xxx.bmodel` and `config` there as well.

#### python demo

Build the library files to generate the `chat.cpython*.so` file, and copy that file to the directory of `pipeline.py`

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

* python demo

``` shell
python3 pipeline.py -m qwen_xxx.bmodel -c config
```
model is the actual storage path of the bmodel; config is the configuration file generated when compiling the model. The demo already includes the qwen3 config.
* If you switch to another model family, such as deepseek-r1-distill-qwen, you need to specify the new config
* qwen3 supports disabling deep thinking by appending '/no_think' after the question; the deepseek-distill version does not support this feature
* Programmatic (non-interactive) mode: specify a one-shot question with `-p/--prompt`; the program runs one inference and exits.

``` shell
python3 pipeline.py -m qwen_xxx.bmodel -c config -p "你好，请简单介绍一下你自己。"
```


#### cpp demo

``` shell
mkdir -p build
cd build
cmake .. && make && cd ..

# how to run
./pipeline -m qwen3_xxx.bmodel -c config
```

Programmatic (non-interactive) mode: specify a one-shot question with `-p/--prompt`; the program runs one inference and exits.

``` shell
./pipeline -m qwen3_xxx.bmodel -c config -p "你好，请简单介绍一下你自己。"
```
