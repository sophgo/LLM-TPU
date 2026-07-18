# Janus-Pro

This project implements the deployment of the large language model [Deepseek-Janus-Pro-7b](https://huggingface.co/deepseek-ai/Janus-Pro-7B) or [Deepseek-Janus-Pro-7b](https://www.modelscope.cn/models/deepseek-ai/Janus-Pro-1B) on BM1684X. The model is converted into a bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed with C++ code to the BM1684X PCIE environment or SoC environment.

The following assumes a PCIE environment by default; if you are using a SoC environment, just follow the prompts.

# Directory Structure
```
.
├── README.md
├── compile
│   ├── compile.sh                          # script used to compile the TPU model
│   ├── export_onnx.py                      # script used to export onnx
│   └── files                               # files used to replace those in the original model
├── python_demo
│   ├── chat.cpp                            # main program file
│   └── pipeline.py                         # execution script for python_demo
├── requirements.txt                        # wheel packages required for environment setup
└── processor_config                        # tokenizer and preprocessing configurations
    ├── special_tokens_map.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ...
```
----------------------------

#  Automated Inference Script

# [Phase 1] Model Compilation

## Notes
* Model compilation must be done inside docker and cannot be performed outside docker. If you do not plan to compile the model, you can also use our precompiled model and jump directly to [Compile the program](#compile-the-program).

### Step 1: Download the model
You can download it from the official HuggingFace or ModelScope sites.
[huggingface](https://huggingface.co/deepseek-ai/Janus-Pro-7B)


### Step 2: Download docker

Download docker and start the container as follows:

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

### Step 3: Download the TPU-MLIR code and compile it

``` shell
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```
* PS: When you re-enter the docker environment and need to compile the model, you must run the above `source ./envsetup.sh` and `./build.sh` in this path before proceeding with subsequent model compilation.

### Step 4: Align the model environment

``` shell
pip install -r requirements.txt
```

### Step 5: Generate the bmodel file

Generate the single-chip model

``` shell
llm_convert.py -m /path/to/Janus-Pro-1B/ -s 710 -q bf16 -g 128 --num_device 1  -c bm1684x  -o janus/ # same as int8
```

----------------------------

# [Phase 2] Executable File Generation

## Compile the program
If you do not plan to compile the model yourself, you can directly use the downloaded model.
```
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/janus-pro-7b_int4_seq2048.bmodel
```
```
python3 -m dfss --url=open@sophgo.com:/share/hengyang/janus-pro-1b_bf16_seq710_bm1684x_1dev_20251021_154608.bmodel(1b model)
```

Run the following compilation (the PCIE version and the SoC version are the same):

```shell
cd python_demo
mkdir build && cd build
cmake .. && make
cp *chat* ..
```

## Model inference
```shell
cd ./python_demo
python3 pipeline.py -m bmodel_path -i image_path -t ../support/processor_config --devid your_devid
```

* Other available parameters can be viewed via `pipeline.py` or by running the following command:
```shell
python3 pipeline.py --help
```
