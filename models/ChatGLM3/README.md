![](./assets/sophgo_chip.png)

# ChatGLM3

This project deploys the large language model [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b) on BM1684X. The model is converted into a bmodel through the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler and deployed to the BM1684X PCIE environment or SoC environment using C++ code.


We wrote an interpretation of `ChatGLM` on Zhihu to help everyone understand the source code:

[ChatGLM2 Flow Analysis and TPU-MLIR Deployment](https://zhuanlan.zhihu.com/p/641975976)


## Development Environment


1. Download docker and start the container as follows:

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
The following assumes that the environment is in the `/workspace` directory of docker.


2. Download `ChatGLM3-6B` from HuggingFace. It is quite large and will take a long time.

``` shell
git lfs install
git clone git@hf.co:THUDM/chatglm3-6b
```

3. Download the `TPU-MLIR` code and compile it (you can also directly download and extract the compiled release package).

``` shell
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```

## Compile the Model

Just use the one-click compilation command when compiling; the generated compilation files are saved in the ./chatglm3 directory.

```shell
llm_convert.py -m /workspace/Chatglm3-6 -s 384 -q w4f16 -g 128 --num_device 1  -c bm1684x  -o chatglm3
```

If you want to perform INT8 or INT4 quantization, run the following command, which finally generates the `chatglm3-6b_int8_1dev.bmodel` or `chatglm3-6b_int4_1dev.bmodel` file, as follows:

```shell
llm_convert.py -m /workspace/Chatglm3-6 -s 384 -q int8 -g 128 --num_device 1  -c bm1684x  -o chatglm3 # or int4
```

If you want to perform 2-chip inference, run the following command, which finally generates the `chatglm3-6b_w4f16_2dev.bmodel` file. The same applies to 4-chip and 8-chip configurations (python_demo currently only supports single-chip):

```shell
llm_convert.py -m /workspace/Chatglm3-6 -s 384 -q w4f16 -g 128 --num_device 2  -c bm1688  -o chatglm3
```

If compilation is inconvenient, you can also directly download the compiled model:
```shell
python3 -m dfss --url=open@sophgo.com:/share/hengyang/chatglm3-6b_w4f16_seq512_bm1684x_1dev_20250630_190644.bmodel
```

## Compile the Program (python_demo Version)

Run the following compilation (the same for the PCIE version and the SoC version):

```shell
cd python_demo
mkdir build
cd build
cmake ..
make
mv chat.cpython-310-x86_64-linux-gnu.so ..
cd ..
```

Run `pipeline.py`:
```shell
source ../../../envsetup.sh
python3 pipeline.py --model_path $PATH_TO_BMODEL --tokenizer_path ../support/token_config/ --devid 0 --generation_mode greedy
```

## Compile the Program (C++ Version)

Run the following compilation (the same for the PCIE version and the SoC version):

```shell
cd demo
mkdir build
cd build
cmake ..
make
```

This compiles and generates the chatglm executable. Put `chatglm` into the demo directory, and specify the number of chips and the bmodel path as follows.
Run `chatglm`, which runs `chatglm3-xxx.bmodel` on a single chip by default:
```shell
./chatglm --model chatglm3-xxx.bmodel --tokenizer ../support/tokenizer.model
```

For 2-chip distributed inference, use the following command (for example, to run on chips 2 and 3; use `bm-smi` to query the chip IDs after running `source /etc/profiel`):
```shell
./chatglm --model chatglm3-xxx.bmodel --devid 2,3 --tokenizer ../support/tokenizer.model
```

## Compile the Program (Python Web Version)

```shell
pip install gradio==3.39.0
cd web_demo
mkdir build
cd build
cmake ..
make -j
```

After a successful compilation, `libtpuchat.so*` will be generated. Specify bmodel\_path, token\_path, device\_id, lib_path (the compiled .so file), and dev_id in web_demo.py.
```python
python web_demo.py --dev 0 --bmodel_path your_bmodel_path
```
Then the web demo will run successfully.

For the SoC environment, refer to the C++ version.

PS: Please use gradio==3.39.0 as much as possible, otherwise various problems will occur!!

## FAQ

#### Where does sentencepiece come from?

The project already contains the compiled version, so there is no need to compile it. If you are curious, refer to the following steps.

Download [sentencepiece](https://github.com/google/sentencepiece) and compile it to get `libsentencepiece.a`.

```shell
git clone git@github.com:google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j
```

If you want to compile for the SoC environment, refer to the demo's compilation method and specify the cross-compiler in the makefile.

#### The demo program cannot run properly

If the demo program cannot run after being copied to the runtime environment, e.g. errors such as interfaces not being found.
The reason is that the libraries in the runtime environment are different. Copy the so files from `./support/lib_pcie` (PCIE) or `./support/lib_soc` (SoC) in the demo to the runtime environment and link against those so files.


## Tool Calling
Reference: [Tool Calling](./tools_using/README.md)