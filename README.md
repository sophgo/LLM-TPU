![](./assets/sophgo_chip.png)

# 介绍

本项目将语言大模型部署到算能BM1684X芯片，在PCIE环境或者SoC环境上都可以顺利运行。

我们所实现的所有模型如下所示：

|Model                |INT4                |INT8                |FP16/BF16           |
|:-                   |:-                  |:-                  |:-                  |
|ChatGLM3-6B          |:white\_check\_mark:|:white\_check\_mark:|:white\_check\_mark:|
|Qwen-7B              |:white\_check\_mark:|:white\_check\_mark:|:white\_check\_mark:|
|Qwen-14B             |:white\_check\_mark:|:white\_check\_mark:|:white\_check\_mark:|
|Llama2-7B            |:white\_check\_mark:|:white\_check\_mark:|:white\_check\_mark:|
|Llama2-13B           |                    |:white\_check\_mark:|:white\_check\_mark:|
|Falcon-40B           |                    |:white\_check\_mark:|:white\_check\_mark:|
|Baichuan2-7B         |                    |:white\_check\_mark:|                    |
|WizardCoder-15B      |:white\_check\_mark:|                    |                    |
|Stable Diffusion     |                    |                    |:white\_check\_mark:|
|Stable Diffusion XL  |                    |                    |:white\_check\_mark:|

如果您感兴趣，也可以联系我们[SOPHGO](https://www.sophgo.com/)。

# Quick Start

如果您手上有1684X芯片的SOC板卡或者PCIE板卡，例如Airbox（SOC），那么可以参考以下步骤跑通大语言模型，这里以Llama2-7B为例。

另外SOC的执行步骤和PCIE的有些区别，PCIE必须要安装docker后才能运行，这里将其分开说明。

### SOC

#### 1. 克隆LLM-TPU项目，并执行run.sh脚本
```
git clone https://github.com/sophgo/LLM-TPU.git
./run.sh --model llama2-7b --arch soc
```

### PCIE

#### 1. 安装docker，并进入docker
```
docker pull sophgo/tpuc_dev:latest
docker run --privileged --name mlir -v /dev:/dev -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash
docker exec -it mlir bash
```

#### 2. 克隆LLM-TPU项目，并执行run.sh脚本
```
git clone https://github.com/sophgo/LLM-TPU.git
./run.sh --model llama2-7b --arch pcie
```

### 效果图
跑通后效果如下图所示

![](./assets/qwen-7b.png)

### Command Table

目前有三个用于演示的模型，全部命令如下表所示

|Model                |INT4                        |INT8                                        |
|:-                   |:-                          |:-                                          |
|ChatGLM3-6B          |./run.sh --model chatglm3-6b |./run.sh --model chatglm3-6b --arch pcie    |
|Llama2-7B            |./run.sh --model llama2-7b --arch soc      |./run.sh --model llama2-7b --arch pcie      |
|Qwen-7B              |./run.sh --model qwen-7b --arch soc        |./run.sh --model qwen-7b --arch pcie        |


# 常见问题

### Q：如果我的Airbox盒子没有联网，那么怎么跑通大语言模型？

A：您可以先在联网的大机器上git clone本项目，之后运行 ./run.sh --model llama2-7b --arch soc 

然后把LLM-TPU的全部文件拷贝到Airbox上，必须要是全部文件，包括LLM-TPU/models和LLM-TPU/deploy

最后再在Airbox上运行 ./run.sh --model llama2-7b --arch soc





