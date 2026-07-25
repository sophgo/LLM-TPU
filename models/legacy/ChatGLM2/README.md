![](./assets/sophgo_chip.png)

# ChatGLM2

This project deploys the large language model [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b) on BM1684X. The model is converted into a bmodel through the [TPU-MLIR](https://github.com/sophgo/tpu-mlir) compiler and deployed to the BM1684X PCIE environment or SoC environment using C++ code.


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


2. Download `ChatGLM2-6B` from HuggingFace. It is quite large and will take a long time.

``` shell
git lfs install
git clone git@hf.co:THUDM/chatglm2-6b
```
Then copy config.json and modeling_chatglm.py from ./models/ChatGLM2/compile/files/chatglm2-6b in this project into the downloaded folder above, replacing the files with the same names (users who need a different sequence length, please refer to [FAQ](#faq); the default sequence length = 512).

3. Download the `TPU-MLIR` code and compile it (you can also directly download and extract the compiled release package).

Since mlir is currently still under maintenance, users compiling the GLM series models please download:
``` shell
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/mlir_club/glm_mlir.tar.gz
tar -xf glm_mlir.tar.gz
source source tpu-mlir_v1.6.45-gdc3e9f6b-20231220/envsetup.sh 
```

After mlir maintenance is completed, you can use the following method:
``` shell
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```

## Compile the Model

1. Export all onnx models. If you are prompted that certain components are missing during the process, just run `pip3 install component`.

``` shell
cd compile
python3 export_onnx.py --model_path your_chatglm2-6b_path
```
At this point, a large number of onnx models are exported to the tmp directory.

2. Compile the onnx models.

TPU-MLIR currently supports F16, INT8, and INT4 quantization for ChatGLM2, and supports multi-chip distributed inference. By default, F16 quantization and single-chip inference are performed, finally generating the `chatglm2-6b_f16_1dev.bmodel` file.

```shell
./compile.sh --name chatglm2-6b --mode inference_mode --num_device device_number
```

Where:
`--name` is the model name, specified here as `chatglm2-6b`;
`--mode` is the data type used for inference. You can choose any of `f16, int8, int4`; the default is `f16`;
`--num_device` is the number of chips used for inference. Please specify it according to the actual devices used; the default is `--num_device 1`.

## Compile the Program (C++ Version)

Run the following compilation (the same for PCIE and SOC):

```shell
cd demo
mkdir build
cd build
cmake ..
make
```

This compiles and generates the chatglm executable. Put `chatglm` into the demo directory, and specify the number of chips and the bmodel path as follows.
Run `chatglm`, which runs `chatglm2-6b_f16_1dev.bmodel` on a single chip by default:
```shell
./chatglm --model chatglm2-6b_f16_1dev.bmodel --tokenizer ../support/tokenizer/tokenizer.model --devid  your_devid
```
Here `--devid` is the ID of the TPU used for inference, which defaults to 0. If you use multi-chip inference (make sure the compiled bmodel is also multi-chip), you can use `,` to add chips, e.g. `--devid 2,3` means using TPU2 and TPU3 for inference.

## Running Result

The following is the running result of INT8 quantization mode on a single chip:

![](./assets/chatglm.jpg)

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


#### What modifications were made to the source code:

Three modifications were made in total:
- Set `seq_length` in the `config.json` file to 512;
- Change the following code in the `modeling_chatglm.py` file:

```python
if attention_mask is not None:
    attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
```

to:

```python
if attention_mask is not None:
    attention_scores = attention_scores + (attention_mask * -10000.0)
```

This modification improves efficiency, since using `masked_fill` is inefficient; on the other hand, there are some bugs when converting `masked_fill` to ONNX.

- Change the following code in the `modeling_chatglm.py` file:

```python
pytorch_major_version = int(torch.__version__.split('.')[0])
if pytorch_major_version >= 2:
```

to:

```python
pytorch_major_version = int(torch.__version__.split('.')[0])
if False:
```

This is because ONNX cannot support the conversion of the `torch.nn.functional.scaled_dot_product_attention` operator.