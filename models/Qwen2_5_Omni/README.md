# Qwen2.5-Omni

This project deploys the multimodal large model [Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B-AWQ) on BM1684X/BM1688. The model is converted into a bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed to a PCIE or SoC environment using C++ code.

This model can be used with images or videos, and videos may contain audio.

The bmodel compilation step can be skipped by downloading directly from the following link:

``` shell
# Does not include the talker
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-omni-7b-awq_w4bf16_seq4096_bm1684x_1dev_no_talker.bmodel 
```

## Compiling the LLM

This section describes how to compile the LLM into a bmodel.

#### 1. Download `Qwen2.5-Omni-7B-AWQ` from HuggingFace

(The model is quite large and will take a long time)

``` shell
# Download the model
git lfs install
git clone git@hf.co:Qwen/Qwen2.5-Omni-7B-AWQ
```

#### 2. Download docker and start the container

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
The following sections assume the environment is in the `/workspace` directory inside docker.

#### 2. Download the `TPU-MLIR` code and compile it

(You can also directly download and extract the precompiled release package)

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  # activate environment variables
./build.sh # compile mlir
```

#### 3. Compile the model to generate the bmodel

``` shell
# If you are prompted about a transformers version issue, run pip3 install transformers -U
llm_convert.py -m /workspace/Qwen2.5-Omni-7B-AWQ  -s 2048 --quantize w4f16  -c bm1684x --out_dir qwen2.5o --max_pixels 672,896
```

## Compiling and running the program (python)

* Environment setup
> (This must be done before running any python_demo)
``` shell
# If your python is not 3.10, refer to "FAQ" to configure the environment
pip3 install torchvision pillow qwen_vl_utils transformers ffmpeg-python -U
```

Compile the library to generate the `chat.cpython*.so` file, then copy it to the directory containing `pipeline.py`

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..

# run demo
python3 pipeline.py -m xxxx.bmodel -c config 
```
`model` is the actual path where the model is stored; `config_path` is the path of the configuration file.



### Getting a "draw.mp4" recognition error?

The error is as follows:
```shell
LibsndfileError: Error opening 'draw.mp4': Format not recognised.
```

Solution:
```shell
sudo apt-get update
sudo apt-get install libsndfile1 ffmpeg --upgrade
pip3 install ffmpeg-python -U
```