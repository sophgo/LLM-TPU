# Qwen2.5-Omni

本工程实现BM1684X/BM1688部署多模态大模型[Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B-AWQ)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到PCIE环境，或者SoC环境。

该模型可以用于图片或者视频，视频中可以带有音频。

如何编译bmodel环节可以省去，直接用以下链接下载：

``` shell
# 不包括talker
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-omni-7b-awq_w4bf16_seq4096_bm1684x_1dev_no_talker.bmodel 
```

## 编译LLM模型

此处介绍如何将LLM编译成bmodel。

#### 1. 从Huggingface下载`Qwen2.5-Omni-7B-AWQ`

(比较大，会花费较长时间)

``` shell
# 下载模型
git lfs install
git clone git@hf.co:Qwen/Qwen2.5-Omni-7B-AWQ
```

#### 2. 下载docker，启动容器

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
后文假定环境都在docker的`/workspace`目录。

#### 2. 下载`TPU-MLIR`代码并编译

(也可以直接下载编译好的release包解压)

``` shell
cd /workspace
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh  #激活环境变量
./build.sh #编译mlir
```

#### 3. 编译模型生成bmodel

``` shell
# 如果有提示transformers版本问题，pip3 install transformers -U
llm_convert.py -m /workspace/Qwen2.5-Omni-7B-AWQ  -s 2048 --quantize w4f16  -c bm1684x --out_dir qwen2.5o --max_pixels 672,896
```

## 编译与运行程序(python)

* 环境准备
> （python_demo运行之前都需要执行这个）
``` shell
# 如果不是python3.10，参考"常见问题"配置环境
pip3 install torchvision pillow qwen_vl_utils transformers ffmpeg-python -U
```

编译库文件，生成`chat.cpython*.so`文件，将该文件拷贝到`pipeline.py`文件目录

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..

# run demo
python3 pipeline.py -m xxxx.bmodel -c config 
```
model为实际的model储存路径；config_path为配置文件路径



### 出现"draw.mp4"识别错误？

错误如下：
```shell
LibsndfileError: Error opening 'draw.mp4': Format not recognised.
```

解决方法：
```shell
sudo apt-get update
sudo apt-get install libsndfile1 ffmpeg --upgrade
pip3 install ffmpeg-python -U
```