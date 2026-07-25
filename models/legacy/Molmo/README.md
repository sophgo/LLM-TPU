# Molmo-7B

This project deploys the large language model [Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924) on BM1684X. The model is converted to a bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler and deployed with C++ code to a BM1684X PCIE or SoC environment.

The following assumes a PCIE environment by default; for a SoC environment, simply follow the prompts.

# Directory description
```
.
├── README.md
├── compile
│   ├── compile.sh                          # script for compiling the TPU model
│   ├── export_onnx.py                      # script for exporting onnx
│   └── files                               # files used to replace those of the original model
├── python_demo
│   ├── chat.cpp                            # main program file
│   └── pipeline.py                         # execution script of python_demo
├── requirements.txt                        # wheel packages to install for environment setup
├── run_demo.sh                             # automated test script
└── processor_config                        # tokenizer and preprocessing configurations
    ├── special_tokens_map.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ...
```
----------------------------

#  Automated inference script



# [Phase 1] Model compilation

## Notes
* Model compilation must be done inside docker; it cannot be done outside docker

### Step 1: Download the model
You can download it from the official HuggingFace or ModelScope pages:
[huggingface](https://huggingface.co/allenai/Molmo-7B-D-0924)
[ModelScope](https://modelscope.cn/models/LLM-Research/Molmo-7B-D-0924/)


### Step 2: Download docker

Download docker and start the container, as follows:

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

### Step 3: Download the TPU-MLIR code and build it

``` shell
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```
* PS: When you re-enter the docker environment and need to compile the model, you must run `source ./envsetup.sh` and `./build.sh` in this path before proceeding with the subsequent model compilation.

### Step 4: Align the model environment

``` shell
pip install -r requirements.txt
cp ./compile/files/Molmo-7B-D-0924/* $your_model_path
```

### Step 5: Generate the onnx files

``` shell
cd compile
python export_onnx.py -m $your_model_path -s 1024 -i 384
```

* PS1: -s is the model sequence length, default 1024. Exporting a model with a length below 1024 is not recommended, because image tokens usually occupy more than 512 of the seq len.
* PS2: -i is the image size, default 384, meaning the input image is 384x384. If the width and height of the input image differ, you can manually modify line 44 of the export_onnx.py script.
* PS3: The image size must be specified. Its purpose is to generate a fake input of the corresponding size saved as weights, which is used to generate the static model later.
* PS4: Steps 3 to 6 can be completed by running run_compile.sh in the compile folder. The specific command is:
``` shell
./run_compile.sh --model_name molmo-7b --seq_length 1024 --model_path your model path --tpu_mlir_path your tpu_mlir path
```
If model_path is not provided, the script will download the model from modelscope; if tpu_mlir_path is not provided, the script will download the corresponding tpu_mlir archive via dfss and extract it.
----------------------------
### Step 6: Generate the bmodel file

Generate a single-chip model

``` shell
./compile.sh --mode int4 --name molmo-7b --seq_length 1024 # same as int8
```
* PS1: Generating the bmodel takes about 3 hours or more. 64 GB of memory and more than 200 GB of disk space are recommended, otherwise OOM or no space left errors are very likely.
* PS2: --name must be specified as molmo-7b
----------------------------

# [Phase 2] Executable generation

## Build the program (Python Demo version)
Run the following build (the PCIE version is the same as the SoC version):

```shell
cd python_demo
mkdir build && cd build
cmake .. && make
cp *chat* ..
```

## Model inference (Python Demo version)
```shell
cd ./python_demo
python3 pipeline.py -m bmodel_path -i image_path -s image_size -t ../processor_config --devid your_devid
```
* Note: image size must be specified as the same image size used when exporting onnx; the pipeline resizes the input image to image size.
* Other available parameters can be viewed in `pipeline.py` or by running the following command:
```shell
python3 pipeline.py --help
```
