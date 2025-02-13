# 大模型精度优化

当采用W4A16或者W8A16效果很不理想时，可以按照本文步骤尝试提升精度。

## 步骤一：使用llmc对大模型进行校准

按照项目[llmc-tpu](https://github.com/sophgo/llmc-tpu)中README的步骤，对大模型进行校准。
最终会生成新的LLM的权重，将该权重覆盖当前大模型权重，然后用torch跑一遍效果，确定效果是正常的。

## 步骤二：重新导出ONNX

按照原有步骤导出ONNX。

* 如果大模型类型是BF16的，请将VIT部分导出成F32；LLM部分可以导F32或者BF16都可以
* 如果大模型类型是F16的，则VIT和LLM都转换成F16，(也可以转成F32)

## 步骤三：转换ONNX

在原有模型compile脚本基础上做以下调整：

### 转换VIT模型

* 如果是转F16类型，需要增加参数`--high_precision`

参考如下：

``` shell
model_deploy.py \
    --mlir vit.mlir \
    --quantize F16 \
    --quant_input \
    --quant_output \
    --high_precision \
    --chip bm1684x \
    --model vit.bmodel
```

* 如果是转换BF16类型，则尝试量化用F16输出用BF16

这一步是为了是VIT用F16推理输出BF16，但是也有可能会出现数值溢出导致乱码；如果会导致乱码，则还是用BF16转换。

参考如下：

``` shell
# 转换后测试模型是否有乱码，没有则OK
model_deploy.py \
    --mlir vit.mlir \
    --quantize F16 \
    --quant_input \
    --quant_output_bf16 \
    --high_precision \
    --chip bm1684x \
    --model vit.bmodel

# 如果有乱码，则按BF16转换
model_deploy.py \
    --mlir vit.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --high_precision \
    --chip bm1684x \
    --model vit.bmodel
```

### 转换Block和BlockCache模型

在原有脚本中增加`--high_precision`参数即可

### 转换lmhead模型

不要用W4A16类型，直接用A16类型(W4BF16则改成BF16;W4F16则改成F16)，如下：

``` shell
model_deploy.py \
    --mlir lm_head.mlir \
    --quantize F16/BF16 \
    --quant_input \
    --chip bm1684x \
    --model lm_head.bmodel
```

### 其他模型

不用修改

## 测试

查看新生成的bmodel在芯片的运行效果