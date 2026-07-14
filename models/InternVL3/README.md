# InternVL3

This project deploys the multimodal large model [InternVL3](https://huggingface.co/OpenGVLab/InternVL3-2B-AWQ) on BM1684X/BM1688. The model is converted into a bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed to a PCIE or SoC environment using C++ code.

## Compile

You can directly download the precompiled model:
``` shell
# =============== 1684x =====================
# InternVL3-8b bm1684x
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/internvl3-8b-awq_w4bf16_seq2048_bm1684x_1dev_20250716_105016.bmodel
# InternVL3-2b bm1684x
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/internvl3-2b-awq_w4bf16_seq2048_bm1684x_1dev_20250716_105401.bmodel

# Advanced 1: supports history context (block with prefill); maximum input length is 2048
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/internvl3-2b-awq_w4bf16_seq8192_bm1684x_1dev_kv_20250723_095246.bmodel
# Advanced 2: dynamic compilation; latency varies with input length
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/internvl3-2b-awq_w4bf16_seq8192_bm1684x_1dev_dyn_20250722_201928.bmodel

# =============== 1688 =====================
# InternVL3-2b bm1688
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/internvl3-2b-awq_w4bf16_seq2048_bm1688_2core_20250716_105519.bmodel
```

#### 1. Download `InternVL3-2B-AWQ` from HuggingFace

(The model is quite large and will take a long time)

``` shell
# Download the model
git lfs install
# For the 2B model:
git clone git@hf.co:OpenGVLab/InternVL3-2B-AWQ

# For the 8B model:
git clone git@hf.co:OpenGVLab/InternVL3-8B-AWQ
```

#### 2. Compile the LLM with tpu-mlir

``` shell
# -c bm1688 is used to compile for the bm1688 chip
llm_convert.py -m /workspace/InternVL3-2B-AWQ -s 2048 -q w4bf16 -c bm1684x --out_dir internvl3_2b
```
After compilation completes, `internvl3-2b-xxx.bmodel` and `config` are generated in the specified directory `internvl3-2b`, where config contains the tokenizer and other original configs.

Adding the --do_sample parameter compiles a sampling model; at runtime, sampling can be performed according to the sampling parameters in generation_config.json under the config path.


## Run
``` shell
cd python_demo
mkdir build && cd build 
cmake .. && make && mv chat.*so ..

# For multi-chip cards, you can use -d $device_id to specify the corresponding chip
# If prompted that a package is not installed, simply pip install the corresponding package
python pipeline.py -m $bmodel_path -c $config_path
```
If --do_sample was enabled during compilation, you can also add --do_sample at runtime to enable sampling mode.

## Advanced usage

### 1. Support for history context

By default, the model does not support history context; you need to add the `--use_history_kv` parameter;
you need to specify the prefill chunk length `--chunk_length`; if not specified, it defaults to 1/4 of seq_length;
the history KV length is fixed at seq_length.

As follows:
``` shell
# If you are prompted about a transformers version issue, run pip3 install transformers -U
llm_convert.py -m /workspace/InternVL3-2B-AWQ -s 8192 -q w4bf16 -c bm1684x --out_dir internvl3_2b_kv --use_history_kv --chunk_length 2048
```


### 2. Support for dynamic compilation

By default, the model is statically compiled: the input is inferred at the specified `seq_length`, and any shortfall is padded with zeros and masked out. Dynamic compilation performs inference dynamically according to the input length, which can reduce latency for short inputs when the input length varies greatly. Simply add `--dynamic` to the command.
```shell
llm_convert.py -m /workspace/InternVL3-2B-AWQ -s 8192 -q w4bf16 -c bm1684x --out_dir internvl3_2b_dyn --dynamic
```
