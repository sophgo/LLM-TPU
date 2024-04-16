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
seq_length=
hidden_size=

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
    --addr_mode)
        addr_mode="$2"
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

if [ "$name" = "llama2-7b" ]; then
  num_layers=32
  hidden_size=4096
  echo "Compile Llama2-7B"
elif [ "$name" = "llama2-13b" ]; then 
  num_layers=40
  hidden_size=5120
  echo "Compile Llama2-13B"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mllama2-7b|llama2-13b\033[0m"
  exit 1
fi

if [ x$mode == x"int8" ] || [ x$mode == x"int4" ]; then
    if [ x$mode == x"int8" ]; then
        quantize_args="--quantize W8F16"
    else
        quantize_args="--quantize W4F16 --q_group_size 64"
    fi
    out_model=$name'_'$mode'.bmodel'
fi

if [ x$num_device != x1 ]; then
    device_args="--num_device $num_device"
    out_model=$name'_'$mode'_'$num_device'dev.bmodel'
else
    out_model=$name'_'$mode'_1dev.bmodel'
fi

if [ x$addr_mode == x"io_alone" ]; then
    addr_args="--addr_mode io_alone"
fi

outdir=${folder}/${mode}_${num_device}/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def ../../embedding.pt \
    --input_shapes [[1,$seq_length]] \
    --input_types "int32" \
    --mlir embedding_${seq_length}.mlir


model_deploy.py \
    --mlir embedding_$seq_length.mlir \
    --quantize F16 \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    $device_args \
    --model embedding_${seq_length}_f16.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def ../../embedding.pt \
    --input_shapes [[1,1]] \
    --input_types "int32" \
    --mlir embedding_1.mlir


model_deploy.py \
    --mlir embedding_1.mlir \
    --quantize F16 \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    $device_args \
    --model embedding_1_f16.bmodel

rm *.npz

models=$models' '$outdir'/embedding_1_f16.bmodel '$outdir'/embedding_'$seq_length'_f16.bmodel '

popd

echo $models

outdir=${folder}/$mode"_"$num_device"dev"/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def ../../lm_head.pt \
    --input_shapes [[1,1,${hidden_size}]] \
    --mlir lm_head.mlir


model_deploy.py \
    --mlir lm_head.mlir \
    $quantize_args \
    --quant_input \
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

for ((i=0; i<$num_layers; i++))
do

model_transform.py \
    --model_name block_$i \
    --model_def ../../block_$i.onnx \
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
# rm ../../block_$i.onnx
# rm ../../block_cache_$i.onnx

models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '

done
popd
echo $models

model_tool --combine $models -o $out_model
