# Qwen2.5

本工程实现BM1684X/BM1688部署大模型[Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到PCIE环境，或者SoC环境。


本文包括如何编译bmodel，和如何在BM1684X/BM1688环境运行bmodel。编译LLM环节可以省去，直接用以下链接下载：

### Qwen2.5系列

``` shell
# qwen2.5-1.5b 2k
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-1.5b_int4_seq2048_1dev.bmodel
# qwen2.5-3b 2k
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-3b-instruct-gptq-int4_w4bf16_seq2048_bm1684x_1dev_20250620_134431.bmodel
# qwen2.5-3b 4k
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-3b_int4_seq4096_1dev.bmodel
# qwen2.5-7b 2k
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-7b-instruct-awq_w4bf16_seq2048_bm1684x_1dev_20250616_191537.bmodel
# qwen2.5-14b 512
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-14b_int4_seq512_1dev.bmodel
```

### QwQ-32B

``` shell
# qwq-32b 2k 2dev
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwq-32b-awq_int4_seq2048_2dev.bmodel
# qwq-32b 2k 4dev
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwq-32b-awq_int4_seq2048_4dev.bmodel
# qwq-32b 16k 4dev
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwq-32b-awq_w4bf16_seq16384_bm1684x_4dev_20250430_163346.bmodel
```

### deepseek-r1-distill-qwen系列

``` shell
# deepseek-r1-distill-qwen-1.5b
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-distill-qwen-1-5b.zip
unzip deepseek-r1-distill-qwen-1-5b.zip

# deepseek-r1-distill-qwen-7b
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-distill-qwen-7b.zip
unzip deepseek-r1-distill-qwen-7b.zip

# deepseek-r1-distill-qwen-14b
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-distill-qwen-14b-seq512.zip
unzip deepseek-r1-distill-qwen-14b-seq512.zip

# deepseek-r1-distill-qwen-32b 多芯
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-distill-qwen-32b-seq2048-4dev.zip
unzip deepseek-r1-distill-qwen-32b-seq2048-4dev.zip

# deepseek-r1-distill-qwen-32b 多芯
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-distill-qwen-32b-seq4096-4dev.zip
unzip deepseek-r1-distill-qwen-32b-seq4096-4dev.zip

```

## 编译LLM模型

``` shell
llm_convert.py -m /workspace/Qwen2.5-3B-Instruct -s 512 --quantize w4bf16 -c bm1684x --out_dir qwen2.5_3b
```
编译完成后，在指定目录`qwen2.5_3b`生成`qwen2.5-3b_xxx.bmodel`和`config`，其中config包含原始配置文件以及分词器等等

另外如果指定的seqlen比较长的话，比如8K，可以指定`--dynamic`编译，首token延时会根据实际长度变化，如下：
``` shell
llm_convert.py -m /workspace/Qwen2.5-3B-Instruct -s 8192 --quantize w4bf16 -c bm1684x --dynamic --out_dir qwen2.5_3b
```

## 编译与运行程序

请将程序拷贝到PCIE环境或者SoC环境后再编译。然后把`bmodel`和`config`文件拷贝过去。

#### python demo

编译库文件，生成`chat.cpython*.so`文件，将该文件拷贝到`pipeline.py`文件目录

``` shell
cd python_demo
mkdir build 
cd build && cmake .. && make && cp *cpython* .. && cd ..
```

* python demo

``` shell
python3 pipeline.py -m qwen_xxx.bmodel -c config 
```
model为实际的bmodel储存路径；config为编译模型时生成的配置文件，demo中存放了qwen2.5的config。
* 如果更换其他系列模型，如deepseek-r1-distill-qwen，需要指定新的config

#### python demo parallel

多芯模型使用该demo，使用方式与单芯一致