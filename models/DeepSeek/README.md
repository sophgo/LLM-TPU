![image](../../assets/sophgo_chip.png)

# DeepSeek

本项目实现BM1684X部署语言大模型[DeepSeek-6.7B](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用python代码将其部署到BM1684X的PCIE环境。

# 目录说明
```
.
├── README.md                            #使用说明
├── compile
│   ├── compile.sh                       #bmodel编译脚本
│   ├── export_onnx.py                   #onnx导出脚本
│   └──files                             #onnx导出所需文件
│       └── modeling_llama.py            #deepseek pytorch模型文件
├── deepseek.py                          #deepseek python代码
├── model                                #bmodel默认目录
├── requirements                         #python wheel包
│   ├── requirements.txt                 #python wheel包依赖
│   └── sophon-3.7.0-py3-none-any.whl    #sophon驱动wheel包
└── token_config                         #deepseek tokenizer配置文件
    ├── generation_config.json
    ├── tokenizer_config.json
    └── tokenizer.json
```
----------------------------

# 【阶段一】模型编译

## 注意点
* 本项目模型编译，安装依赖，运行例程等步骤均在docker中进行

### 步骤一：模型下载
DeepSeek模型在hugging face上完全开源，供用户下载使用。请根据官网下载步骤进行模型与权重的下载。
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct
```

### 步骤二：下载docker

下载docker，启动容器，如下：

``` shell
docker pull sophgo/tpuc_dev:latest

# deepseek is just an example, you can set your own name
docker run --restart always -td --privileged -v /opt:/opt -v $PWD:/workspace --name deepseek sophgo/tpuc_dev:latest bash
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

``` shell
pushd requirements
pip3 install -r requirements.txt
popd
cp compile/files/modeling_llama.py /usr/local/lib/python3.10/dist-packages/transformers/models/llama/
```

* PS：不一定是/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py这个路径，建议替换前先pip3 show transformers查看一下

### 步骤五：生成onnx文件

``` shell
cd compile
python export_onnx.py --model_path your_model_path --seq_length 512
```

* PS1：your_model_path 指的是原模型下载后的地址, 如:"../deepseek-coder-6.7b-instruct/"
* PS2：默认导出sequence length为512的模型

### 步骤六：生成bmodel文件

``` shell
./compile.sh --mode int4 --name deepseek-6.7b # same as int8
#when finished
mv deepseek-6.7b_int4_1dev.bmodel ../model/
```

* PS1：编译完成后最终会在compile路径下生成名为deepseek-6.7b_{X}_1dev.bmodel,其中X为`compile.sh`时选择的`mode`的数据类型
* PS2：生成bmodel耗时大概3小时以上，建议64G内存以及200G以上硬盘空间，不然很可能OOM或者no space left
* PS3：--name必须为deepseek-6.7b

----------------------------

# 阶段二：运行例程

执行如下命令运行例程（以int4为例）

``` shell
# default args: --bmodel ./model/deepseek-6.7b_int4_1dev.bmodel --token ./token_config/ --dev_id 0
python3 deepseek.py
```


