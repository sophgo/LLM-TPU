# Qwen3-ASR

This project deploys the speech recognition model [Qwen3-ASR](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B) on BM1684X/BM1688. The model is converted into a bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed to a PCIE or SoC environment.

Model download links:
```bash
# BM1684X 1.7B 
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-asr-1.7b_bf16_seq512_bm1684x_1dev_static_20260408_094600.bmodel
# BM1688 0.6B
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-asr-0.6b_bf16_seq256_bm1688_2core_static_20260415_173617.bmodel
```

## Compile the model
This section describes how to compile the LLM into a bmodel.

#### 1. Download `Qwen3-ASR-1.7B` from ModelScope

(The model is quite large and will take a long time.)

``` shell
# Download the 1.7B model
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir Qwen3-ASR-1.7B

# If you want to use the 0.6B model:
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir Qwen3-ASR-0.6B
```

#### 2. Download docker and start the container

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
The following sections assume the environment is in the `/workspace` directory inside docker.

#### 2. Download and extract the precompiled `TPU-MLIR` release package


``` shell
cd /workspace
# Download the precompiled release package, which supports Qwen3-ASR
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Qwen/qwen3_asr/tpu-mlir_v1.27.beta.0-79-g8bd0e0c1b-20260414.tar.gz
tar zxf tpu-mlir_v1.27.beta.0-79-g8bd0e0c1b-20260414.tar.gz
cd tpu-mlir_v1.27.beta.0-79-g8bd0e0c1b-20260414
source ./envsetup.sh  # activate environment variables
```

#### 3. Compile the model to generate the bmodel

``` shell
# If you are prompted about a transformers/torch version issue, run pip3 install torch==2.4.1 transformers==4.57.6 qwen_asr -U
# Here max_input_length specifies the maximum input length; if not specified, it defaults to the length given by -s
llm_convert.py -m /workspace/Qwen3-ASR-1.7B  -s 512 --max_input_length 256  --quantize bf16  -c bm1684x --out_dir qwen3_asr --qwen_asr
```
After compilation completes, `qwen3-asr-xxx.bmodel` and `config` are generated in the specified directory `qwen3_asr`


## Compiling and running the program (python)

* Environment setup
> This must be done before running python_demo
> Python version 3.10 or above is required. If not met, please refer to the [python3.10 installation](https://github.com/sophgo/sophon-demo/blob/release/docs/FAQ.md#13-se7%E5%AE%89%E8%A3%85python310) documentation
``` shell
sudo apt-get update
sudo apt-get install pybind11-dev

pip3 install torch==2.4.1 transformers==4.57.6 qwen_asr
```

Compile the library to generate the `chat.cpython*.so` file, then copy it to the directory containing `pipeline.py`

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..

# run demo
python3 pipeline.py -m xxxx.bmodel -c ../config 
```
`model` is the actual path where the model is stored; `config_path` is the path of the configuration file.

* Note:
> The demo is non-streaming
> 1 second of audio takes up about 13 tokens