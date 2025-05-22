#!/bin/bash
set -x
models=
mode="w4bf16"
name=""
seq_length=4096
chip="bm1684x"
folder="tmp"
quantize_args="--quantize W4BF16"
addr_args="--addr_mode io_alone"

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    --mode)
        mode="$2"
        shift 2
        ;;
    --name)
        name="$2"
        shift 2
        ;;
    --seq_length)
        seq_length="$2"
        shift 2
        ;;
    *)
        echo "Invalid option: $key" >&2
        exit 1
        ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1
        ;;
    esac
done

if [[ -z "$seq_length" ]]; then
    echo "Error: --seq_length is required." >&2
    exit 1
fi

if [ "$name" = "internvl3-14b" ]; then
  num_layers=48
  hidden_size=5120
elif [ "$name" = "internvl3-8b" ]; then
  num_layers=28
  hidden_size=3584
elif [ "$name" = "internvl3-2b" ]; then
  num_layers=28
  hidden_size=1536
elif [ "$name" = "internvl3-1b" ]; then
  num_layers=24
  hidden_size=896
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31minternvl3-14b|internvl3-8b|internvl3-2b|internvl3-1b\033[0m"
  exit 1
fi

if [ x$mode == x"w8bf16" ]; then
    quantize_args="--quantize W8BF16"
elif [ x$mode == x"f16" ]; then
    quantize_args="--quantize BF16"
elif [ x$mode == x"w4bf16" ]; then
    quantize_args="--quantize W4BF16 --q_group_size 64"
else
    echo "Error, unknown quantize mode"
    exit 1
fi


outdir=${folder}/bmodel
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def ../onnx/embedding.pt \
    --input_shapes [[1,${seq_length}]] \
    --input_types "int32" \
    --mlir embedding.mlir

model_deploy.py \
    --mlir embedding.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    --model embedding.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def ../onnx/embedding.pt \
    --input_shapes [[1,1]] \
    --input_types "int32" \
    --mlir embedding_cache.mlir

model_deploy.py \
    --mlir embedding_cache.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    --model embedding_cache.bmodel

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '
echo $models

model_transform.py \
    --model_name lm_head \
    --model_def ../onnx/lm_head.pt \
    --input_shapes [[1,${hidden_size}]] \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    $quantize_args \
    --quant_input \
    --chip bm1684x \
    --model lm_head.bmodel

models=${models}${outdir}'/lm_head.bmodel '
echo $models

process_block()
{
    i=$1
    model_transform.py \
        --model_name block_$i \
        --model_def ../onnx/block_$i.onnx \
        --mlir block_$i.mlir

    model_deploy.py \
        --mlir block_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $device_args \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def ../onnx/block_cache_$i.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $device_args \
        $addr_args \
        --model block_cache_$i.bmodel
}

for ((i=0; i<$num_layers; i++)); do
    process_block $i &
    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '
    sleep 45
done

rm -f *.npz *.onnx
echo $models

model_transform.py \
  --model_name vit \
  --model_def ../onnx/vit/vit.onnx \
  --mlir vit.mlir 

model_deploy.py \
  --mlir vit.mlir \
  --quantize BF16 \
  --quant_output \
  --chip bm1684x \
  --model vit.bmodel

models=${models}${outdir}'/vit.bmodel '
echo $models

rm -f *.npz *.onnx
popd

out_model=${name}_${mode}_seq${seq_length}_${chip}.bmodel
model_tool --combine $models -o $out_model
model_tool --info $out_model > ${folder}'/model.log'