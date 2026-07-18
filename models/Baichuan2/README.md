![image](../../assets/sophgo_chip.png)

# Baichuan2-TPU

This project deploys the large language model [Baichuan2-7B](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat) on BM1684X. The model is converted into a bmodel through the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler and deployed to the BM1684X PCIE environment or SoC environment using C++ code.

The following assumes the PCIE environment by default; for the SoC environment, just follow the prompts.

# Directory Structure
```
.
├── README.md                           #Usage instructions
├── requirements.txt                    #Required python wheel packages
├── compile
│   ├── compile.sh                      #Script used to compile the TPU model
│   ├── export_onnx.py                  #Script used to export onnx
│   ├── torch_inference.py              #torch inference script
│   └── files
│       └── Baichuan2-7B                #Backup of files that replace the corresponding files in Baichuan2-7B-chat
│           ├── config.json
│           └── modeling_baichuan.py
├── demo                                #Baichuan2 C++ code files
│   ├── CMakeLists.txt
│   └── demo.cpp                        #Main program
├── src                                 #Compilation dependency libraries
│   ├── include
│   ├── lib_pcie
│   └── lib_soc
├── model                               #Model files (bmodel needs to be downloaded)
│   ├── baichuan2-7b-test_int8.bmodel
│   └── tokenizer.model
└── web_demo                            #web demo, provides a web chat example
    ├── chat.cpp
    ├── chat.py
    ├── CMakeLists.txt
    └── web_demo.py
```
----------------------------

# [Stage 1] Model Compilation

## Notes
* Model compilation must be done inside docker; it cannot be performed outside docker.

### Step 1: Download the Model
The Baichuan2 model is fully open source on HuggingFace for users to download and use. Please download the model and weights following the official download steps.
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat
```

### Step 2: Download docker

Download docker and start the container as follows:

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

### Step 3: Download and Compile the TPU-MLIR Code

``` shell
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```
* PS: When you re-enter the docker environment and need to compile a model, you must run `source ./envsetup.sh` and `./build.sh` under this path before proceeding with the subsequent model compilation.

### Step 4: Download This Project and Install requirements.txt
Download transformers, sentencepiece, Baichuan2-TPU, and the .bin model from Baidu Netdisk, then replace the modeling_baichuan.py in transformers.

``` shell
git clone https://github.com/sophgo/Baichuan2-TPU.git
cd Baichuan2
pip3 install -r requirements.txt
```

### Step 5: Replace modeling_baichuan.py, Modify config.json, and Generate the onnx Files
Modify max_position_embeddings and model_max_length in the config.json file of the Baichuan2-7B-chat project from 4096 to 512.

``` shell
cd compile
cp files/Baichuan2-7B/modeling_baichuan.py $BAICHUAN2_PATH
cp files/Baichuan2-7B/config.json $BAICHUAN2_PATH
python3 export_onnx.py --model_path $BAICHUAN2_PATH
```

* PS1: your_model_path refers to the location where the original model was downloaded, e.g. "../../torch2onnx/Baichuan2-7B-Chat". You can choose to use either the 7b model or the 13b model as needed.
* PS2: If you want to debug instead of generating all the onnx models at once, you can change num_layers on line 240 to 1 and use function comparison to check whether a single block matches.

### Step 6: Generate the bmodel File

Generate the model:

``` shell
./compile.sh --mode int8
mv baichuan2-7b_int8_1dev.bmodel ../model
```

* PS1: After compilation, a file named baichuan2-{X}b_{Y}_{Z}dev.bmodel will be generated under the Baichuan2-TPU/compile path, where X is 7 or 13, Y is the data type of the `mode` selected when running `compile.sh`, and Z is the number of chips used for inference (if num_device is not specified, the {Z}dev part will be omitted).
* PS2: Generating the bmodel takes about 3 hours or more. It is recommended to have 64 GB of memory and more than 200 GB of disk space, otherwise OOM or no space left errors are likely.
* PS3: The currently provided lib_pcie and lib_soc only contain the single-chip dynamic libraries; the multi-chip part will be updated later.

----------------------------

# Stage 2: Executable Generation (Can Be Skipped)

## Preparation
* bmodel preparation: After Stage 1 you will have the compiled bmodel file [you can also use the ready-made compiled bmodel file we provide]. Download it as follows:
```shell
cd Baichuan2-TPU/model
pip3 install dfss
# baichuan2-7B
python3 -m dfss --url=open@sophgo.com:sophon-demo/baichuan2/baichuan2-7b-test_int8.bmodel
```
This gives you the compiled int8 single-chip bmodel file.

## Compile the Program (C++ Version)

Run the following compilation; the PCIE version is the default:

```shell
cd Baichuan2-TPU/demo
mkdir build
cd build
cmake ..
make
```

For the SoC version, there are two compilation methods:

Method 1: Copy the demo directory directly to the SoC environment and compile with the steps above (recommended).

Method 2: Cross-compile in docker as follows:

```shell
wget https://releases.linaro.org/components/toolchain/binaries/7.5-2019.12/aarch64-linux-gnu/gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz
tar -xvf gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu.tar.xz
mv gcc-linaro-7.5.0-2019.12-x86_64_aarch64-linux-gnu /opt/aarch64-linux-gnu-7.5.0
cd Baichuan2-TPU/demo
mkdir build
cd build
cmake .. -DTARGET_ARCH=soc # soc has only one chip, so multi-chip compilation is not supported
make -j
```

This compiles and generates the Baichuan2 executable.

Run `baichuan2`:
```shell
./baichuan2 --model ../model/baichuan2-7b-test_int8.bmodel --dev dev_id
```

## Compile the Program (Python Web Version) [Single-Chip]

```shell
pip3 install gradio==3.39.0
cd Baichuan2-TPU/web_demo
mkdir build
cd build
cmake ..
make -j
```

After a successful compilation, `libtpuchat.so*` will be generated in the `build` folder. At this point, you can specify bmodel\_path, token\_path, device\_id, lib_path (the compiled `libtpuchat.so*` file, which is under `./build` by default), and dev_id in web_demo.py.
```python
python3 web_demo.py
```
Then the web demo will run successfully.
* PS: As long as the user does not modify the storage paths of token\_path and lib\_path above, only bmodel\_path needs to be specified to run the program.

For the SoC environment, refer to the C++ version.

* PS: Please use gradio==3.39.0 as much as possible, otherwise various problems will occur!!

# FAQ
* Please adjust NUM_LAYERS in `demo/chat` or `web_demo/chat.cpp` according to the actual number of blocks. By default it uses Baichuan2-7B (NUM_LAYERS=32).
