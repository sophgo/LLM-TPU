![image](./assets/sophgo_chip.png)

# Llama2

This project demonstrates deploying the large language model [Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) on BM1684X. The model is converted into a bmodel using the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler and deployed to a BM1684X PCIE environment or SoC environment using C++ code.

The following assumes a PCIE environment by default; for an SoC environment, just follow the prompts.

# Directory description
```
.
├── README.md                            # usage instructions
├── requirements.txt                     # required python wheel packages
├── demo                                 # Llama2 c++ code files
│   ├── CMakeLists.txt
│   └── demo.cpp                         # main program
├── web_demo                             # Llama2 web demo code files
│   ├── CMakeLists.txt
│   ├── chat.cpp                         # cpp main program
│   ├── chat.py                          # python main program after pybind
│   └── web_demo.py                      # gradio python interface code
├── compile
│   ├── compile.sh                       # script for compiling the TPU model
│   └── export_onnx.py                   # script for exporting onnx
│       └── files                        # files used to replace the original model files
└── support
    ├── include                          # library files required for compilation
    ├── lib_pcie                         # header files required for compiling the PCIE version
    ├── lib_soc                          # header files required for compiling the SOC version
    └── tokenizer.model                  # tokenizer model
```
----------------------------

# [Stage 1] Model compilation

## Notes
* Model compilation must be done inside docker; it cannot be done outside docker

### Step 1: Download the model
Although the Llama2 model is open source for commercial use, downloading the model requires submitting a usage application to Meta, so when testing the model you can use the model we have already downloaded
```bash
pip3 install dfss
# llama2-7B
python3 -m dfss --url=open@sophgo.com:sophon-demo/Llama2/llama2-7b-torch.zip
unzip llama2-7b-torch.zip

# llama2-13B
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/llama2-13b-torch.zip
unzip llama2-13b-torch.zip
```

### Step 2: Download docker

Download docker and start the container, as follows:

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
* PS: When re-entering the docker environment and needing to compile the model, you must run the above `source ./envsetup.sh` and `./build.sh` in this path before proceeding with subsequent model compilation.

### Step 4: Align the model environment

``` shell
pip install -r requirements.txt
cp ./compile/files/llama-2-7b-chat-hf/modeling_llama.py /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py
```

* PS: The path is not necessarily /usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py; it is recommended to run pip show transformers to check before replacing.

### Step 5: Generate the onnx files

``` shell
cd compile
python export_onnx.py --model_path your_model_path --seq_length 512
```

* PS1: your_model_path refers to the path of the downloaded original model, e.g. "../../torch2onnx/llama-2-7b-chat-hf". You can choose to use the 7b model or the 13b model as needed.
* PS2: If you want to debug instead of generating all the onnx models at once, you can change num_layers on line 240 to 1, and use the function on line 233 to compare whether a single block matches.
* PS3: By default, a model with a sequence length of 512 is exported.

### Step 6: Generate the bmodel file

Generate a single-chip model

``` shell
./compile.sh --mode int8 --name llama2-7b # same as int4
```

Generate a dual-chip model

``` shell
./compile.sh --mode int8 --num_device 2 --name llama2-7b # same as int4
```

* PS1: After compilation, a file named llama2-{X}b_{Y}_{Z}dev.bmodel will be generated in the compile path, where X is 7 or 13, Y is the data type of the `mode` selected in `compile.sh`, and Z is the number of chips used for inference (if num_device is not specified, the {Z}dev part is omitted).
* PS2: Generating the bmodel takes more than about 3 hours; 64G of memory and more than 200G of disk space are recommended, otherwise OOM or no space left errors are very likely.
* PS3: If you want to compile llama2-7b, --name must be llama2-7b; if you want to compile llama2-13b, --name must be llama2-13b.
* PS4: The currently provided lib_pcie and lib_soc only contain single-chip dynamic libraries; the multi-chip part will be updated later.

----------------------------

# Stage 2: Executable generation

## Compile the program (C++ version)

Run the following compilation (the PCIE version is the same as the SoC version):

```shell
cd demo
mkdir build
cd build
cmake ..
make
```

Compilation generates the llama2 executable. Put `llama2` in the demo directory, and specify the chip number (only chip 0 is used by default) and the bmodel path as follows. Run `llama2`:
```shell
./llama2 --model your_llama2_bmodel_path --tokenizer ../support/tokenizer.model --dev dev_id
```

For dual-chip distributed inference, use the following command (for example, to run on chips 2 and 3; use `bm-smi` after `source /etc/profiel` to query the chip IDs; the libsophon driver must have been installed beforehand):
```shell
./llama2 --model your_llama2_bmodel_path --tokenizer ../support/tokenizer.mode --devid 2,3
```
* PS: Do not use multiple chips to run inference on a compiled single-chip model. You can tell whether it is a multi-chip model from the compiled bmodel name; for example, `llama2-7b_int8_2dev.bmodel` is a model that can run on dual chips. A dual-chip model can run on a single chip.

## Compile the program (Python Web version) [single chip]

```shell
pip install gradio==3.39.0
cd web_demo
mkdir build
cd build
cmake ..
make -j
```

After successful compilation, `libtpuchat.so*` will be generated in the `build` folder. At this point, you can specify bmodel\_path, token\_path, device\_id, lib_path (the compiled `libtpuchat.so*` file, default path is under `./build`), and dev_id in web_demo.py.
```python
python web_demo.py --dev 0 --bmodel_path your_bmodel_path
```
The web demo will then run successfully.
* PS: As long as the user does not modify the storage paths of the above token\_path and lib\_path, only bmodel\_path needs to be specified to run the program.

For an SoC environment, refer to the C++ version.

* PS: Try to download gradio==3.39.0, otherwise various problems will occur!!

# FAQ

### How to run the Llama2-13B 6-chip model

First, follow the version check in [this link](https://github.com/sophgo/LLM-TPU/tree/main) to check whether the sophon-driver version is 0.5.0; if it is 0.4.9, it will hang.

```shell
cd /workspace/LLM-TPU/models/Llama2/demo
mkdir build && cd build
cmake .. && make && cp llama2 .. && cd ..

python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama2-13b_int4_6dev.bmodel

./llama2 --model llama2-13b_int4_6dev.bmodel --tokenizer ../support/token_config/tokenizer.model  --devid 0,1,2,3,4,5


python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama2-13b_int8_6dev.bmodel

./llama2 --model llama2-13b_int8_6dev.bmodel --tokenizer ../support/token_config/tokenizer.model  --devid 0,1,2,3,4,5
```

