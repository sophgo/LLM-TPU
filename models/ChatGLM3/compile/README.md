# Command

## Compile bmodel

```shell
pushd /path_to/tpu-mlir
source envsetup.sh
popd
```

### compile basic bmodel
```shell
llm_convert.py -m /workspace/Chatglm3-6 -s 384 -q f16 -g 128 --num_device 1  -c bm1684x  -o chatglm3
```

若想进行INT8或INT4量化，则执行以下命令，最终生成`chatglm3-6b_int8_1dev.bmodel`或`chatglm3-6b_int4_1dev.bmodel`文件，如下命令：

```shell
llm_convert.py -m /workspace/Chatglm3-6 -s 384 -q $quant_mode -g 128 --num_device 1  -c bm1684x  -o chatglm3
```
