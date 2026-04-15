# Qwen3-ASR

本工程实现BM1684X/BM1688部署语音识别模型[Qwen3-ASR](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并将其部署到PCIE环境，或者SoC环境。

模型下载链接：
```bash
# BM1684X 1.7B 
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-asr-1.7b_bf16_seq512_bm1684x_1dev_static_20260408_094600.bmodel
```

## 编译模型
此处介绍如何将LLM编译成bmodel。

#### 1. 从ModelScope下载`Qwen3-ASR-1.7B`

(比较大，会花费较长时间。)

``` shell
# 下载1.7B模型
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir Qwen3-ASR-1.7B

# 如果想用0.6B模型，如下：
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir Qwen3-ASR-0.6B
```

#### 2. 下载docker，启动容器

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
后文假定环境都在docker的`/workspace`目录。

#### 2. 下载`TPU-MLIR`编译好的release包解压


``` shell
cd /workspace
# 下载编译好的release包，支持Qwen3-ASR
python3 -m dfss --url=open@sophgo.com:/sophon-demo/Qwen/qwen3_asr/tpu-mlir_v1.27.beta.0-79-g8bd0e0c1b-20260414.tar.gz
tar zxf tpu-mlir_v1.27.beta.0-79-g8bd0e0c1b-20260414.tar.gz
cd tpu-mlir_v1.27.beta.0-79-g8bd0e0c1b-20260414
source ./envsetup.sh  #激活环境变量
```

#### 3. 编译模型生成bmodel

``` shell
# 如果有提示transformers/torch版本问题，pip3 install torch==2.4.1 transformers qwen_asr -U
# 这里max_input_length指定最大输入长度，如果不指定则为-s指定的长度
llm_convert.py -m /workspace/Qwen3-ASR-1.7B  -s 512 --max_input_length 256  --quantize bf16  -c bm1684x --out_dir qwen3_asr --qwen_asr
```
编译完成后，在指定目录`qwen3_asr`生成`qwen3-asr-xxx.bmodel`和`config`


## 编译与运行程序(python)

* 环境准备
> python_demo运行之前都需要执行这个
> python版本需要3.10及以上。若不满足，请参考[python3.10安装](https://github.com/sophgo/sophon-demo/blob/release/docs/FAQ.md#13-se7%E5%AE%89%E8%A3%85python310)文档
``` shell
sudo apt-get update
sudo apt-get install pybind11-dev

pip3 install torch==2.4.1 transformers qwen_asr
```

编译库文件，生成`chat.cpython*.so`文件，将该文件拷贝到`pipeline.py`文件目录

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..

# run demo
python3 pipeline.py -m xxxx.bmodel -c ../config 
```
model为实际的model储存路径；config_path为配置文件路径。