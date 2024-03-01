![](./assets/tpumlir.png)

# Qwen-TPU

本工程实现BM1684X部署语言大模型[Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)。通过[TPU-MLIR](https://github.com/sophgo/tpu-mlir)编译器将模型转换成bmodel，并采用c++代码将其部署到BM1684X的PCIE环境，或者SoC环境。

* 本工程也支持[Qwen-14-Chat](https://huggingface.co/Qwen/Qwen-14B-Chat)，操作方法与`Qwen-7B-Chat`一致。
* 本工程也支持[Qwen-1_8-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat)，操作方法与`Qwen-7B-Chat`一致。

## 开发环境准备

### 1. 下载本项目`Qwen-TPU`

下载本项目，并导出所有的ONNX（其中需要将本项目`files`路径下的`config.json`和`modeling_qwen.py`文件替换到原模型的文件夹下，如下：
``` shell
git clone git@github.com:sophgo/LLM-TPU.git

pushd Qwen-TPU
git submodule update --init
popd
```

因为我们采用BF16格式导出ONNX，需要您的环境上带有CUDA。默认x86不支持BF16。14B或1_8B模型需要将`export`指定到对应路径,同时export_onnx.py中的模型路径也许做对应的修改

### 2. 下载pytorch.bin模型(以`Qwen-7B-Chat`为例)(可跳过)

如果你没有nvidia的环境，可以跳过这一步，但必须要执行第四步，下载onnx文件

``` shell
git lfs install
git clone git@hf.co:Qwen/Qwen-7B-Chat

pushd Qwen-TPU/compile
cp files/Qwen-7B-Chat/* ../../Qwen-7B-Chat
export PYTHONPATH=$PWD/../Qwen-7B-Chat:$PYTHONPATH

pip install transformers_stream_generator einops tiktoken
python3 export_onnx.py --model_path ../Qwen-7B-Chat

popd
```

该工程比较大，会花较长时间。
并将本项目下的`files/Qwen-7B-Chat`中的文件替换至`Qwen-7B-Chat`下的对应文件。

### 3. 下载docker，启动容器

``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234 is just an example, you can set your own name
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
```
后文假定环境都在docker的`/workspace`目录。

### 4. 下载onnx模型

由于pytorch.bin转onnx这一步需要nvidia的环境，你也可以直接下载我们转好的模型

``` shell
cd Qwen-TPU/compile
pip3 install dfss
python3 -m dfss --url=open@sophgo.com:/LLM/qwen_8k.zip
unzip qwen_8k.zip
```

### 5. 下载`TPU-MLIR`代码并编译

(也可以直接下载编译好的release包解压)

``` shell
git clone git@github.com:sophgo/tpu-mlir.git
cd tpu-mlir
source ./envsetup.sh
./build.sh
```

## 编译模型

注意此时在Docker环境workspace目录。

目前TPU-MLIR支持对`Qwen-7B`进行BF16、INT8和INT4量化，且支持多芯分布式推理，默认情况下会进行INT8量化和单芯推理，最终生成`qwen-7b_int8.bmodel`文件。

```shell
./compile.sh --name qwen-7b
```

若要编译int4，或者bf16版本，则加入`--model`参数。如下转int4，最终生成`qwen-7b_int4.bmodel`：

```shell
./compile.sh --mode int4 --name qwen-7b
```

若想进行2芯推理，则执行以下命令，最终生成`qwen-7b_int8_2dev.bmodel`文件，4芯8芯同理：

```shell
./compile.sh --num_device 2 --name qwen-7b
```

## 编译程序(C++版本)

执行如下编译 (注意如果是SoC版本，需要把demo目录拷贝到SoC环境编译)：

```shell
cd Qwen-TPU/demo
mkdir build
cd build
cmake ..
make
```

编译生成qwen可执行程序，将`qwen`、`qwen-7b_int8.bmodel`和`qwen.tiktoken`拷贝到同一个目录下就可以执行了。
(`qwen.tiktoken`来自[Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat))。

## 运行`qwen`

### a. 命令行交互
- 单芯推理：使用如下命令。
```shell
./qwen --model qwen-7b_int8.bmodel
```

- 多芯分布式推理：如果是2芯分布式推理，使用如下命令(比如指定在2号和3号芯片上运行, 用`bm-smi`查询芯片id号)：
```shell
./qwen --model qwen-7b_int8_2dev.bmodel --devid 2,3
```

#### 运行效果

以下为单芯片下INT8量化模式的运行效果：

![](./assets/qwen.jpg)


### b. 调用API
1. 执行`pip3 install sse_starlette`，安装sse_starlette。
2. 在`api.py`中，对`create_app()`中初始化QwenChat的参数（devid、bmodel路径和tiktoken路径）作相应修改。
3. 执行`python3 api.py`。稍作等待，启动成功终端显示如图，可通过8000端口访问服务。
![](assets/api_init.png)
4. 发请求到`[设备ip]:8000/chat/completions`，请求格式为
```json
{
    "question": "你知道深圳的历史吗?写一段小作文介绍下。",
    "history": ["Hello, how are you?", "I am fine, thank you."],
    "stream": false
}
```
示例如下：
![](assets/api_test_example.png)


## 常见问题

### demo程序无法正常运行

如果demo程序拷贝到运行环境提示无法运行，比如接口找不到等等错误。
原因是运行环境的库有所不同，将demo中的`lib_pcie`（PCIE）或者 `lib_soc`(SoC)里面的so文件拷贝到运行环境，链接到里面的so即可。

### tiktoken是如何用C++支持的

tiktoken官方没有C++版本，只有python版本。
本工程使用[QwenLM/qwen.cpp](https://github.com/QwenLM/qwen.cpp)中的tiktoken处理代码。

### 如果编译其他seq_length的模型

将Qwen模型中的config.json中`seq_length`改成对应想要的长度即可

### Qwen-7B-Chat做了哪些修改

只对`config.json`和`modeling_qwen.py`做了部分调整。

#### 1. 调整`config.json`文件中参数配置

```json
  "bf16": true,
  "max_position_embeddings": 512,
  "seq_length": 512,
```

我们采用bf16导出ONNX模型，原因是该模型是通过bf16训练的。用F32也可以，但是这样ONNX体积太大。

#### 2. 对`modeling_qwen.py`文件代码做调整

1) 第一点修改如下（这是因为TORCH2的算子转ONNX会失败）：

    ``` python
    # SUPPORT_TORCH2 = hasattr(torch, '__version__') and int(torch.__version__.split(".")[0]) >= 2
    SUPPORT_TORCH2 = False
    ```

2) 第二点修改如下（这是因为转ONNX，提示Shape推导失败）：

    ```python
    # attn_weights = attn_weights / torch.full(
    #     [],
    #     size_temp ** 0.5,
    #     dtype=attn_weights.dtype,
    #     device=attn_weights.device,
    # )
    attn_weights = attn_weights / (size_temp ** 0.5)
    ```

3) 第三点修改如下（这段代码全部注释掉，是因为可以直接采用`attention_mask`，避免复杂逻辑，提升性能）：

    ```python
    # if self.use_cache_quantization:
    #     query_length, key_length = query.size(-2), key[0].size(-2)
    # else:
    #     query_length, key_length = query.size(-2), key.size(-2)
    # causal_mask = registered_causal_mask[
    #     :, :, key_length - query_length : key_length, :key_length
    # ]
    # mask_value = torch.finfo(attn_weights.dtype).min
    # mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
    #     attn_weights.device
    # )
    # attn_weights = torch.where(
    #     causal_mask, attn_weights.to(attn_weights.dtype), mask_value
    # )
    ```

4) 第四点修改如下（同上原因）：

    ``` python
    # query_length, key_length = query.size(-2), key.size(-2)
    # causal_mask = registered_causal_mask[
    #     :, :, key_length - query_length : key_length, :key_length
    # ]
    # mask_value = torch.finfo(attn_weights.dtype).min
    # mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(
    #     attn_weights.device
    # )
    # attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    ```

5) 第五点修改，将如下代码移至`if layer_past is not None:`之前：

    ``` python
    if use_cache:
        present = (key, value)
    else:
        present = None
    ```

    这是因为kv cache只用输出1个单位就可以了，不用全部输出。提升效率。
