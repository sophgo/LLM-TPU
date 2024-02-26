![](./assets/sophgo_chip.png)

# 介绍

本项目将语言大模型部署到算能BM1684X芯片，在PCIE环境或者SoC环境上都可以顺利运行。

另外本项目还给出了三个模型用于演示，如下表所示，如果您手上有SOC板卡 / Airbox（SOC） / PCIE板卡，也可以参考 Quick Start 这一章节跑通大语言模型。

|Model                |SOC                                        |PCIE                                        |
|:-                   |:-                                         |:-                                          |
|ChatGLM3-6B          |./run.sh --model chatglm3-6b --arch soc    |./run.sh --model chatglm3-6b --arch pcie    |
|Llama2-7B            |./run.sh --model llama2-7b --arch soc      |./run.sh --model llama2-7b --arch pcie      |
|Qwen-7B              |./run.sh --model qwen-7b --arch soc        |./run.sh --model qwen-7b --arch pcie        |

上面只是用于演示的模型，除此之外，我们还实现了Falcon-40B，WizardCoder，Qwen-14B，Llama2-14B，Stable Diffusion等等，如果您感兴趣，也可以联系我们[SOPHGO](https://www.sophgo.com/)。

# Quick Start

如果您手上有SOC板卡 / Airbox（SOC） / PCIE板卡，那么可以参考以下步骤跑通大语言模型，这里以Llama2-7B为例。

另外SOC的执行步骤和PCIE的有些区别，PCIE必须要安装docker后才能运行，这里将其分开说明。

### SOC

#### 1. 克隆LLM-TPU项目，并执行run.sh脚本
```
git clone https://github.com/sophgo/LLM-TPU.git
./run.sh --model llama2-7b --arch pcie
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


# 常见问题

### Q：如果我的Airbox盒子没有联网，那么怎么跑通大语言模型？

A：你可以先在联网的大机器上git clone本项目，之后运行 ./run.sh --model llama2-7b --arch soc 

然后把LLM-TPU的全部文件拷贝到Airbox上，必须要是全部文件，包括LLM-TPU/models和LLM-TPU/deploy

最后再在Airbox上运行 ./run.sh --model llama2-7b --arch soc





