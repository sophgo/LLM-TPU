# InternVL3

本工程实现BM1684X/BM1688部署多模态大模型[InternVL3](https://huggingface.co/OpenGVLab/InternVL3-2B-AWQ)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

## 编译

可以直接下载编译好的模型:
``` shell
# InternVL3-8b bm1684x
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/internvl3-8b_w4bf16_seq4096_bm1684x.bmodel
# InternVL3-2b bm1684x
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/internvl3-2b_w4bf16_seq4096_bm1684x.bmodel

# InternVL3-2b bm1688
python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/internvl3-2b-awq_w4bf16_seq2048_bm1688_2core.bmodel
```

#### 1. 从Huggingface下载`InternVL3-2B-AWQ`

(比较大，会花费较长时间)

``` shell
# 下载模型
git lfs install
# 如果是2B，则如下：
git clone git@hf.co:OpenGVLab/InternVL3-2B-AWQ

# 如果是8B，则如下：
git clone git@hf.co:OpenGVLab/InternVL3-8B-AWQ
```

#### 2. 使用tpu-mlir编译LLM

``` shell
# -c bm1688 用于编译bm1688芯片
llm_convert.py -m /workspace/InternVL3-2B-AWQ -s 2048 -q w4bf16 -c bm1684x --out_dir internvl3-2b
```
编译完成后，在指定目录`internvl3-2b`生成`internvl3-2b-xxx.bmodel`和`config`，其中config包含tokenizer和其他原始config。

添加--do_sample参数可编译采样模型，运行时可根据config路径中的generation_config.json采样参数进行采样。


## 运行
``` shell
cd python_demo
mkdir build && cd build 
cmake .. && make && mv chat.*so ..

# 对于有多芯卡，可以用-d $device_id，指定对应的芯片
# 如果有提示未安装包，用pip install对应的安装包即可
python pipeline.py -m $bmodel_path -c $config_path
```
如果在编译时打开了--do_sample，运行时也可以选择加上--do_sample，开启采样模式。

