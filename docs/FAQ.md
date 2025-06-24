# 常见问题

## 目录
- [常见问题](#常见问题)
  - [Q1：如果我的BM1684X环境没有联网，那么怎么跑通大语言模型？](#q1如果我的bm1684x环境没有联网那么怎么跑通大语言模型)
  - [Q2：为什么在PCIE模式下，我在docker里运行以后第一次输出会出现如下的warning？](#q2为什么在pcie模式下我在docker里运行以后第一次输出会出现如下的warning)
  - [Q3：推理出来精度异常，输出全是“！”](#q3推理出来精度异常输出全是)
  - [Q4：执行python_demo时报这个错 `ValueError: vector::_M_default_append`](#q4执行python_demo时报这个错-valueerror-vectorm_default_append)
  - [Q5：跑Qwen1_5的时候遇到 `RuntimeError: The size of tensor a (16) must match the size of tensor b (512) at non-singleton dimension 1`](#q5跑qwen1_5的时候遇到-runtimeerror-the-size-of-tensor-a-16-must-match-the-size-of-tensor-b-512-at-non-singleton-dimension-1)
  - [Q6：跑Qwen1_5、Qwen2的时候遇到 `ValueError: Tokenizer class Qwen2Tokenizer does not exist or is not currently imported.`](#q6跑qwen1_5qwen2的时候遇到-valueerror-tokenizer-class-qwen2tokenizer-does-not-exist-or-is-not-currently-imported)
  - [Q7：`FATAL:BMRT_ASSERT: _kernel_modules[core_id]`](#q7fatalbmrt_assert-_kernel_modulescore_id)
  - [Q8：遇到生僻字或者emoji会出这种问号](#q8遇到生僻字或者emoji会出这种问号)
  - [Q9：`FATAL:BMRT ASERT: (shape count * data type size) <= get device size`](#q9fatalbmrt-asert-shape-count--data-type-size--get-device-size)
  - [Q10：`[bmlib memoryllerrorl bm alloc gmem failed`](#q10bmlib-memoryllerrorl-bm-alloc-gmem-failed)
  - [Q11：`[a53lite runtimellerror] get function send api error, ret2`](#q11a53lite-runtimellerror-get-function-send-api-error-ret2)
  - [Q12：`The repository for /path contains custom code which must be executed to correctly load the model`](#q12the-repository-for-path-contains-custom-code-which-must-be-executed-to-correctly-load-the-model)
  - [Q13：输出乱码](#q13输出乱码)
  - [Q14：`[BMRT][fix gdma addr:488] FATAL:gdma dst shouldn't be coeff`](#q14bmrtfix-gdma-addr488-fatalgdma-dst-shouldnt-be-coeff)
  - [Q15：`ImportError: /lib/x86_64-linux-gnu/libc.so.6: version 'GLIBC 2.32' not found`](#q15importerror-libx86_64-linux-gnulibcso6-version-glibc-232-not-found)
  - [Q16：！！！感叹号问题](#q16感叹号问题)
  - [Q17：`AttributeError: 'LlamaTokenizerFast' object has no attribute 'apply_chat_template'`](#q17attributeerror-llamatokenizerfast-object-has-no-attribute-apply_chat_template)
  - [Q18：`torch.onnx.errors.UnsupportedOperatorError`](#q18torchonnxerrorsunsupportedoperatorerror)
  - [Q19：`NameError: name 'Extension' is not defined`](#q19nameerror-name-extension-is-not-defined)
  - [Q20：`unzip: short read`](#q20unzip-short-read)

---

## 常见问题

### Q1：如果我的BM1684X环境没有联网，那么怎么跑通大语言模型？

**问题描述**  
BM1684X环境没有联网时，如何运行大语言模型？

**解决方案**  
1. 在联网的大机器上运行以下命令：
   ```bash
   git clone https://github.com/sophgo/LLM-TPU.git
   ./run.sh --model llama2-7b
   ```
2. 将 `LLM-TPU` 的全部文件拷贝到Airbox上，包括 `LLM-TPU/models` 和 `LLM-TPU/deploy`。
3. 在Airbox上运行以下命令：
   ```bash
   ./run.sh --model llama2-7b
   ```

---

### Q2：为什么在PCIE模式下，我在docker里运行以后第一次输出会出现如下的warning？

**问题描述**  
运行时出现以下warning：
```shell
[a53lite_runtime][error] open file /opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so error!!
[a53lite_runtime][error] /workspace/libsophon/bmlib/src/a53lite_api.cpp 488: load file failed!
bm_module is null!
```

**解决方案**  

这是由于SDK版本过低导致，请到官网下载最新SDK:
https://developer.sophgo.com/site/index/material/all/all.html

### Q3：推理出来精度异常，输出全是“！”

**问题描述**  
推理结果异常，输出全是感叹号“！”。

**解决方案**  
可能原因及对应解决方式如下：

**方式一：** TPU电压过低导致降频，执行以下命令降频：
```bash
echo "setr tpll_clock 750000000" > /sys/kernel/debug/top/clock
echo "setr mpll_clock 1800000000" > /sys/kernel/debug/top/clock
echo "setr vpll_clock 100000000"> /sys/kernel/debug/top/clock
```

**方式二：** 断电几分钟后清理缓存：
```bash
echo 3 > /proc/sys/vm/drop_caches
```

**方式三：** 数据格式问题，切换 `fp16` 和 `bf16` 格式：
1. 如果 `compile.sh` 中的 `quantize_args` 为：
   ```bash
   quantize_args="--quantize W4BF16 --q_group_size 64"
   ```
   则修改为：
   ```bash
   quantize_args="--quantize W4F16 --q_group_size 64"
   ```
2. 如果原来是 `W4F16`，则改为 `W4BF16`。

---

### Q4：执行python_demo时报这个错 `ValueError: vector::_M_default_append`

**问题描述**  
执行 `python_demo` 时出现以下错误：
```shell
ValueError: vector::_M_default_append
```

**解决方案**  
修改 `CMakeLists.txt` 文件，将第一行改为：
```cmake
cmake_minimum_required(VERSION 3.10)
```

---

### Q5：跑Qwen1_5的时候遇到 `RuntimeError: The size of tensor a (16) must match the size of tensor b (512) at non-singleton dimension 1`

**问题描述**  
运行Qwen1_5时，出现以下错误：
```shell
RuntimeError: The size of tensor a (16) must match the size of tensor b (512) at non-singleton dimension 1
```

**解决方案**  
是由于 `torch` 版本不对导致的。推荐使用以下版本：
```bash
pip3 install torch==2.0.1+cpu torchvision==0.15.2 -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

---

### Q6：跑Qwen1_5、Qwen2的时候遇到 `ValueError: Tokenizer class Qwen2Tokenizer does not exist or is not currently imported.`

**问题描述**  
运行Qwen1_5或Qwen2时，报以下错误：
```shell
ValueError: Tokenizer class Qwen2Tokenizer does not exist or is not currently imported.
```

**解决方案**  
安装正确的 `transformers` 版本：
```bash
pip3 install transformers==4.37.0
```

---

### Q7：`FATAL:BMRT_ASSERT: _kernel_modules[core_id]`

**问题描述**  
运行时出现以下错误：
```shell
[BMRT][preload_funcs:2352] FATAL:BMRT_ASSERT: _kernel_modules[core_id]
```

**解决方案**  
芯片挂死，尝试以下方法：
1. 更换一颗芯片。
2. 或者重新启动服务器（请谨慎操作）：
   ```bash
   sudo reboot
   ```

### Q8：遇到生僻字或者emoji会出这种问号

**问题描述**  
在大模型中，有些不常用的字是由两个token组成，拆开解码时会出现问号。例如：
```shell
Question: my name is lao wang

Answer: Nice to meet you, ���! I'm Llama3, your helpful AI assistant. How can I assist you today? Do you have any questions, topics you'd like to discuss, or tasks you'd like to accomplish? I'm here to help!
```
问题可能出现在解码过程中。

**解决方案**  
参考以下[提交](https://github.com/sophgo/LLM-TPU/commit/eec3c0edc33daf109d6682d5dc156ad63c83a6a1)进行修复。

---

### Q9：`FATAL:BMRT ASERT: (shape count * data type size) <= get device size`

**问题描述**  
运行时出现以下报错：
```shell
FATAL:BMRT ASERT: (shape count * data type size) <= get device size(shape count:8388608 * data type size:2) shouldn't larger than mem get device size:
```
这是因为模型编译有问题，长度没有对齐。

**解决方案**  
1. 使用以下命令查看模型shape情况：
   ```shell
   model_tool --info xxx.bmodel
   ```
   示例输出如下：
   ![](./pics/Q9_2.png)

2. 根据输出信息检查并调整模型编译参数。

---

### Q10：`[bmlib memoryllerrorl bm alloc gmem failed`

**问题描述**  
运行时出现以下报错：
```shell
[bmlib memoryllerrorl bm alloc gmem failed, dev id = 12, size = 0x25180000
```
通常是因为模型过大或者卡的空间不足。

**解决方案**  
**方式一：**  
1. 开一个新的终端，在运行模型时使用以下命令观察显存使用情况：
   ```shell
   bm-smi
   ```
2. 如果显存使用远小于14000MB，可以申请更多空间：
   ```shell
   ./memory_edit.sh -p
   ./memory_edit.sh -c -npu 7168 -vpu 3072 -vpp 4096
   ```
   参考[这篇文章](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html)。

**方式二：**  
如果模型太大，尝试使用 `fp16/bf16` 编译，或者使用 `w4fp16/w4bf16` 编译：
```shell
./compile.sh --mode int4 --name qwen2-7b --addr_mode io_alone
```

---

### Q11：`[a53lite runtimellerror] get function send api error, ret2`

**问题描述**  
运行时出现以下错误：
![](./pics/Q11.png)  
可能的原因包括：
- 模型有问题。
- driver版本过低（如0.4.8或0.4.9）。
- `libsophon` 版本不是最新的。
- 输入长度超过最大限制。

**解决方案**  
**方式一：**  
1. 使用以下命令查看driver版本：
   ```shell
   bm-smi
   ```
2. 如果driver版本过低，更新到最新版本：
   - 从官网上下载：[SDK-24.04.01](https://developer.sophgo.com/site/index/material/all/all.html)。
   - 或者使用 `dfss` 下载：
     ```shell
     pip3 install dfss
     python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/drivers/libsophon-0611deb.tar.gz
     tar -xzf libsophon-0611deb.tar.gz
     cd libsophon-0611deb
     sudo apt remove sophon-driver sophon-libsophon
     sudo dpkg -i *.deb
     ```

---

### Q12：`The repository for /path contains custom code which must be executed to correctly load the model`

**问题描述**  
运行时出现以下错误：
![](./pics/Q12.png)  
这是因为加载模型时需要信任远程代码。

**解决方案**  
在代码中添加 `trust_remote_code=True` 参数：
```python
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
```

---

### Q13：输出乱码

**问题描述**  
运行时输出乱码，如下图所示：
![](./pics/Q13.png)  
可能的原因包括：
- 模型有问题。
- `libsophon` 版本不是最新的。
- `prompt` 格式有问题。

**解决方案**  
1. 更新 `tpu-mlir` 到最新版本，同时将driver设置为0.5.1及以上，并更新 `libsophon`，参考Q11。
2. 检查 `prompt` 格式：
   - 使用 `transformers` 库的 `model.chat` 函数测试输出的tokens和 `system prompt`。
   - 在 `pipeline.py` 中设置断点，检查tokens是否对齐。

---

### Q14：`[BMRT][fix gdma addr:488] FATAL:gdma dst shouldn't be coeff`

**问题描述**  
运行时出现以下错误：
![](./pics/Q14.png)  
这是因为 `libsophon` 和 driver 版本过旧。

**解决方案**  
更新 `libsophon` 和 driver 到2024年6月30日之后的版本，参考Q11。

---

### Q15：`ImportError: /lib/x86_64-linux-gnu/libc.so.6: version 'GLIBC 2.32' not found`

**问题描述**  
运行时出现以下错误：
![](./pics/Q15.png)  
这是因为编译环境与运行环境不一致。

**解决方案**  
1. 删除 `build` 文件夹并重新编译：
   ```shell
   rm -rf build && mkdir build
   cd build && cmake .. && make
   ```
2. 如果是PCIE服务器，请在docker环境中编译so文件。

---

### Q16：！！！感叹号问题

**问题描述**  
推理结果中出现大量感叹号“！！！”，可能是中间计算为 `nan` 导致的。  
- 如果结果全为 `nan`（如“！！！”），则是 `forward_first` 出错。
- 如果只有部分为 `nan`（如“我！！！！”），则是 `forward_next` 出错。

**解决方案**  
1. 使用 `gdb` 定位问题：
   - 在 `embedding` 前后、`block` 前后、`lmhead` 前后，用 `dump_fp16_tensor` 检查输入输出。
   - 确保输入与 `export_onnx` 的 `test_net_with_mask` 对齐。
2. 如果某个block出错（如block11），尝试以下方法：
   - **开比对：** 用 `dump_net` 函数导出block11的输入，用 `model_deploy` 进行比对。
   - **置零：** 将非真实值置零，用 `empty` 函数处理。
3. 检查 `libsophon` 版本、驱动版本、`tpu-mlir` 版本是否匹配。

---

### Q17：`AttributeError: 'LlamaTokenizerFast' object has no attribute 'apply_chat_template'`

**问题描述**  
运行时出现以下错误：
```shell
AttributeError: 'LlamaTokenizerFast' object has no attribute 'apply_chat_template'
```
这是因为 `transformers` 版本过低。

**解决方案**  
更新 `transformers`：
```shell
pip3 install git+https://github.com/huggingface/transformers
```

---

### Q18：`torch.onnx.errors.UnsupportedOperatorError`

**问题描述**  
运行时出现以下错误：
```shell
torch.onnx.errors.UnsupportedOperatorError: Exporting the operator 'aten::_convolution_mode' to ONNX opset version 15 is not supported.
```
通常出现在多模态模型的 `Conv` 算子中，原因是 `torch` 版本过低。

**解决方案**  
更新 `torch` 版本：
```shell
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

---

### Q19：`NameError: name 'Extension' is not defined`

**问题描述**  
运行时出现以下错误：
```bash
NameError: name 'Extension' is not defined
```
这是因为缺少 `Jinja2` 依赖。

**解决方案**  
安装 `Jinja2`：
```shell
pip3 install Jinja2
```

---

### Q20：`unzip: short read`

**问题描述**  
运行时出现以下错误：
```shell
unzip: short read
```

**解决方案**  
安装 `zip` 工具：
```shell
sudo apt install zip
```