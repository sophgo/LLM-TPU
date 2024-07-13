#!/bin/bash
set -ex
models=
mode="int8"
folder="tmp"
num_device=1
mode_args=""
device_args=""
quantize_args="--quantize W8BF16"
addr_args=""
dyn_args=""
name=""
num_layers=
hidden_size=
seq_length=
out_model=$name.bmodel
share_length=
unshare_length=
hidden_size=
dynamic=0

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
    --share_length)
        share_length="$2"
        shift 2
        ;;
    --unshare_length)
        unshare_length="$2"
        shift 2
        ;;
    --dynamic)
        dynamic="$2"
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

if [[ -z "$share_length" ]]; then
    echo "Error: --share_length is required." >&2
    exit 1
fi

if [ "$name" = "qwen2-7b" ]; then
  num_layers=28
  hidden_size=3584
  echo "Compile Qwen2-7B"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mqwen2-7b\033[0m"
  exit 1
fi

if [ x$mode == x"int8" ]; then
    quantize_args="--quantize W8BF16"
elif [ x$mode == x"bf16" ]; then
    quantize_args="--quantize BF16"
elif [ x$mode == x"int4" ]; then
    quantize_args="--quantize W4BF16 --q_group_size 64"
else
    echo "Error, unknown quantize mode"
    exit 1
fi

if [ x$num_device != x1 ]; then
    device_args="--num_device $num_device"
    out_model=${name}_${mode}_share${share_length}_unshare${unshare_length}_${num_device}dev.bmodel
else
    out_model=${name}_${mode}_share${share_length}_unshare${unshare_length}_1dev.bmodel
fi

if [ x$dynamic == x1 ]; then
    dyn_args="--dynamic"
    out_model=${name}_${mode}_share${share_length}_unshare${unshare_length}_${num_device}dev_dyn.bmodel
fi

if [ x$addr_mode == x"io_alone" ]; then
    addr_args="--addr_mode io_alone"
fi

outdir=${folder}/$mode"_"$num_device"dev"/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def ../../onnx/embedding.pt \
    --input_shapes [[1,${share_length}]] \
    --input_types "int32" \
    --mlir embedding.mlir

model_deploy.py \
    --mlir embedding.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    $device_args \
    --model embedding.bmodel

model_transform.py \
    --model_name embedding_unshare \
    --model_def ../../onnx/embedding.pt \
    --input_shapes [[1,${unshare_length}]] \
    --input_types "int32" \
    --mlir embedding_unshare.mlir

model_deploy.py \
    --mlir embedding_unshare.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    $device_args \
    --model embedding_unshare.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def ../../onnx/embedding.pt \
    --input_shapes [[1,1]] \
    --input_types "int32" \
    --mlir embedding_cache.mlir

model_deploy.py \
    --mlir embedding_cache.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    $device_args \
    --model embedding_cache.bmodel

rm -f *.npz

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_unshare.bmodel '$outdir'/embedding_cache.bmodel '

popd

echo $models

outdir=${folder}/$mode"_"$num_device"dev"/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def ../../onnx/lm_head.pt \
    --input_shapes [[1,${hidden_size}]] \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    $quantize_args \
    --quant_input \
    --chip bm1684x \
    $device_args \
    --model lm_head.bmodel


model_transform.py \
    --model_name greedy_head \
    --model_def ../../onnx/greedy_head.onnx \
    --mlir greedy_head.mlir

model_deploy.py \
    --mlir greedy_head.mlir \
    --chip bm1684x \
    --model greedy_head.bmodel


model_transform.py \
    --model_name penalty_sample_head \
    --model_def ../../onnx/penalty_sample_head.onnx \
    --mlir penalty_sample_head.mlir

model_deploy.py \
    --mlir penalty_sample_head.mlir \
    --chip bm1684x \
    --model penalty_sample_head.bmodel

rm *.npz

models=${models}${outdir}'/lm_head.bmodel '$outdir'/greedy_head.bmodel '$outdir'/penalty_sample_head.bmodel '

popd
echo $models

outdir=${folder}/$mode"_"$num_device"dev"/block
mkdir -p $outdir
pushd $outdir

# Function to process each block in parallel
process_block() {
    i=$1

    model_transform.py \
        --model_name block_$i \
        --model_def ../../onnx/block_$i.onnx \
        --mlir block_$i.mlir

    model_deploy.py \
        --mlir block_$i.mlir \
        ${quantize_args} \
        --quant_input \
        --quant_output \
        $dyn_args \
        --chip bm1684x \
        $device_args \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_unshare_$i \
        --model_def ../../onnx/block_unshare_$i.onnx \
        --mlir block_unshare_$i.mlir

    model_deploy.py \
        --mlir block_unshare_$i.mlir \
        ${quantize_args} \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $device_args \
        --model block_unshare_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def ../../onnx/block_cache_$i.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        ${quantize_args} \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $device_args \
        $addr_args \
        --model block_cache_$i.bmodel
}

# Process each block in parallel
for ((i=0; i<$num_layers; i++)); do
    process_block $i &
    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_unshare_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '
    sleep 30
done
rm -f *.npz
popd
echo $models

wait  # Wait for all background processes to finish

model_tool --combine $models -o $out_model
