#!/bin/bash
set -ex
models=
mode="f16"
num_device=1
quantize_args="--quantize F16"
device_args=""
out_model=wizardcoder-15B.bmodel

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
        quantize_args="--quantize W4F16 --q_group_size 64"
    fi
    out_model="wizardcoder-15B_$mode.bmodel"
fi

if [ x$num_device != x1 ]; then
    device_args="--num_device $num_device"
    out_model='wizardcoder-15B_'$mode'_'$num_device'dev.bmodel'
fi

outdir=tmp/embedding
mkdir -p $outdir
pushd $outdir

seqlen=512

model_transform.py \
    --model_name embedding \
    --model_def ../onnx/embedding.onnx \
    --input_shapes [[1,$seqlen],[1,$seqlen]] \
    --mlir embedding_${seqlen}.mlir


model_deploy.py \
    --mlir embedding_$seqlen.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model embedding_${seqlen}_f16.bmodel

model_transform.py \
    --model_name embedding \
    --model_def ../onnx/embedding.onnx \
    --input_shapes [[1,1],[1,1]] \
    --mlir embedding_1.mlir


model_deploy.py \
    --mlir embedding_1.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model embedding_1_f16.bmodel

rm *.npz

models=$models' '$outdir'/embedding_1_f16.bmodel '$outdir'/embedding_'$seqlen'_f16.bmodel '

popd

echo $models

outdir=tmp/$mode"_"$num_device"dev"/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def ../../onnx/lm_head.onnx \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    --quantize W8F16 \
    --chip bm1684x \
    $device_args \
    --model lm_head.bmodel

rm *.npz

models=${models}${outdir}'/lm_head.bmodel '
popd

echo $models

outdir=tmp/$mode"_"$num_device"dev"/glm_block
mkdir -p $outdir

pushd $outdir
mkdir -p $outdir

for i in {0..39}
do

model_transform.py \
    --model_name block_$i \
    --model_def ../../onnx/block_$i.onnx \
    --mlir block_$i.mlir

model_deploy.py \
    --mlir block_$i.mlir \
    $quantize_args \
    --chip bm1684x \
    $device_args \
    --model block_$i.bmodel

model_transform.py \
    --model_name block_cache_$i \
    --model_def ../../onnx/block_cache_$i.onnx \
    --mlir block_cache_$i.mlir

model_deploy.py \
    --mlir block_cache_$i.mlir \
    $quantize_args \
    --chip bm1684x \
    $device_args \
    --model block_cache_$i.bmodel

rm *.npz

models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '

done
popd
echo $models


model_tool --combine $models -o $out_model
