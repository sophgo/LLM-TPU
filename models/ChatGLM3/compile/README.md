# Command

## Export onnx

```shell
python export_onnx.py --model_path your_torch_path --device cpu
```

## Compile bmodel

```shell
pushd /path_to/tpu-mlir
source envsetup.sh
popd
```

### compile basic bmodel
```shell
./compile.sh --name chatglm3-6b
```

若想进行INT8或INT4量化，则执行以下命令，最终生成`chatglm3-6b_int8_1dev.bmodel`或`chatglm3-6b_int4_1dev.bmodel`文件，如下命令：

```shell
./compile.sh --mode int8 --name chatglm3-6b # or int4
```
