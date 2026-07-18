# Llama3.2

This project implements the deployment of the large language model [Llama-3.2-11B-Vision-Instruct](https://www.modelscope.cn/models/LLM-Research/Llama-3.2-11B-Vision-Instruct) on BM1684X. The model is converted to bmodel through the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler, and deployed to the PCIE environment or the SoC environment of BM1684X using c++ code.

The following assumes the PCIE environment by default; if you are in the SoC environment, just follow the prompts.

# [Stage 1] Model compilation

## Notes
* Model compilation must be done inside docker; it cannot be done outside docker

### Step 1: Model download
Although the Llama3 model is commercially open source, downloading the model requires submitting a usage application to Meta. Therefore, when testing the model, you can download it by referring to the [model weights provided by ModelScope](https://www.modelscope.cn/models/LLM-Research/Llama-3.2-11B-Vision-Instruct), or apply for the Meta License through HuggingFace to download.


### Step 2: Download docker

Download docker and start the container as follows:

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

### Step 3: Download TPU-MLIR code and compile

``` shell
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```
* PS: When re-entering the docker environment and needing to compile a model, you must execute the above `source ./envsetup.sh` and `./build.sh` in this path before you can complete the subsequent model compilation.

### Step 4: Set up the model environment

Use the llm_convert.py script to compile the bmodel; the compilation results are saved in the mllama3_2 folder

``` shell
pip install -r requirements.txt
llm_convert.py -m /workspace/Llama-3.2-11B-Vision-Instruct/ -s 512 -q w4bf16 -g 64 --num_device 1  -c bm1684x  -o mllama3_2/
```
----------------------------

# [Stage 2] Executable file generation

## Compile the program (Python Demo version)
Execute the following compilation (same for the PCIE version and the SoC version):

```shell
cd python_demo
mkdir build
cd build
cmake ..
make
cp *chat* ..
```

## Model inference (Python Demo version)
```shell
cd ./python_demo
python3 pipeline.py -m your_model_path -i your_image_path -t ../token_config --devid your_devid
```
Other available parameters can be viewed through `pipeline.py` or by executing the following command
```shell
python3 pipeline.py --help
```
