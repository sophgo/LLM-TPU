# Llama3.2

本项目实现BM1684X部署语言大模型[Llama-3.2-11B-Vision-Instruct](https://www.modelscope.cn/models/LLM-Research/Llama-3.2-11B-Vision-Instruct)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

下文中默认是PCIE环境；如果是SoC环境，按提示操作即可。

# 【阶段一】模型编译

## 注意点
* 模型编译必须要在docker内完成，无法在docker外操作

### 步骤一：模型下载
虽然Llama3模型允许商业开源，但是模型下载需要想Meta提交使用申请，因此测试模型时可以参考[ModelScope提供的模型权重](https://www.modelscope.cn/models/LLM-Research/Llama-3.2-11B-Vision-Instruct)进行下载，或者通过Huggingface申请Meta License进行下载。


### 步骤二：下载docker

下载docker，启动容器，如下：

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```

### 步骤三：下载TPU-MLIR代码并编译

``` shell
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```
* PS：重新进入docker环境并且需要编译模型时，必须在此路径下执行上述`source ./envsetup.sh` 和 `./build.sh`才能完成后续模型编译。

### 步骤四：对齐模型环境

使用 llm_convert.py 脚本编译bmodel，编译结果保存于 mllama3_2 文件夹

``` shell
pip install -r requirements.txt
llm_convert.py -m /workspace/Llama-3.2-11B-Vision-Instruct/ -s 512 -q w4bf16 -g 64 --num_device 1  -c bm1684x  -o mllama3_2/
```
----------------------------

# 【阶段二】可执行文件生成

## 编译程序(Python Demo版本)
执行如下编译，(PCIE版本与SoC版本相同)：

```shell
cd python_demo
mkdir build
cd build
cmake ..
make
cp *chat* ..
```

## 模型推理(Python Demo版本)
```shell
cd ./python_demo
python3 pipeline.py -m your_model_path -i your_image_path -t ../token_config --devid your_devid
```
其它可用参数可以通过`pipeline.py` 或者执行如下命令进行查看 
```shell
python3 pipeline.py --help
```
