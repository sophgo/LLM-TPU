#!/bin/bash
set -ex
models=
mode="f16"
folder="tmp"
num_device=1
mode_args=""
device_args=""
quantize_args="--quantize F16"
name=""
num_layers=
out_model=$name.bmodel

if [ -z "$name" ]; then
    name="llama2-7b"
    echo "Compile Llama2-7B"
else
    name="llama2-13b"
    echo "Compile Llama2-13B"
fi

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --mode)
            mode="$2"
            shift 2
            ;;
        --num_device)
            num_device="$2"
            shift 2
            ;;
        --name)
            name="$2"
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

if [ x$mode == x"int8" ] || [ x$mode == x"int4" ]; then
    if [ x$mode == x"int8" ]; then
        quantize_args="--quantize W8F16"
    else
        quantize_args="--quantize W4BF16 --q_group_size 64"
    fi
    out_model=$name'_'$mode'.bmodel'
fi

if [ x$name == x"llama2-7b" ] || [ x$name == x"llama2-13b" ]; then
    if [ x$name == x"llama2-7b" ]; then
        num_layers=0
    else
        num_layers=39
    fi
fi

if [ x$num_device != x1 ]; then
    device_args="--num_device $num_device"
    out_model=$name'_'$mode'_'$num_device'dev.bmodel'
else
    out_model=$name'_'$mode'_1dev.bmodel'
fi

outdir=${folder}/embedding
mkdir -p $outdir
pushd $outdir

seqlen=512
model_transform.py \
    --model_name embedding \
    --model_def ../onnx/embedding.onnx \
    --mlir embedding.mlir

model_deploy.py \
    --mlir embedding.mlir \
    --quantize F16 \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    $device_args \
    --model embedding.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def ../onnx/embedding.onnx \
    --input_shapes [[1]] \
    --mlir embedding_cache.mlir

model_deploy.py \
    --mlir embedding_cache.mlir \
    --quantize F16 \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    $device_args \
    --model embedding_cache.bmodel

rm *.npz

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '

popd

echo $models

outdir=${folder}/$mode"_"$num_device"dev"/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def ../../onnx/lm_head.onnx \
    --mlir lm_head.mlir


model_deploy.py \
    --mlir lm_head.mlir \
    $quantize_args \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    --model lm_head.bmodel

rm *.npz

models=${models}${outdir}'/lm_head.bmodel '
popd

echo $models

outdir=${folder}/$mode"_"$num_device"dev"/block
mkdir -p $outdir

pushd $outdir
mkdir -p $outdir

for ((i=0; i<=$num_layers; i++))
do

model_transform.py \
    --model_name block_$i \
    --model_def ../../onnx/block_$i.onnx \
    --mlir block_$i.mlir

model_deploy.py \
    --mlir block_$i.mlir \
    $quantize_args \
    --chip bm1684x \
    --quant_input \
    --quant_output \
    $device_args \
    --model block_$i.bmodel

model_transform.py \
    --model_name block_cache_$i \
    --model_def ../../block_cache_${i}.onnx \
    --mlir block_cache_$i.mlir

model_deploy.py \
    --mlir block_cache_$i.mlir \
    $quantize_args \
    --chip bm1684x \
    --quant_input \
    --quant_output \
    $device_args \
    --model block_cache_$i.bmodel

rm *.npz

models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '

done
popd
echo $models

model_tool --combine $models -o $out_model
