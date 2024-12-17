#!/bin/bash
set -ex
models=
folder="tmp"
device_args=""
quantize_args="--quantize W4BF16"
addr_args="--addr_mode io_alone"
name=""
num_layers=
out_model=$name.bmodel
hidden_size=
mode="int4"

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
    --addr_mode)
        addr_mode="$2"
        shift 2
        ;;
    --seq_length)
        seq_length="$2"
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

if [[ -z "$seq_length" ]]; then
    echo "Error: --seq_length is required." >&2
    exit 1
fi

if [ "$name" = "vila1.5-3b" ]; then
  num_layers=32
  hidden_size=2560
  echo "Compile VILA1.5-3B"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mvila1.5-3b\033[0m"
  exit 1
fi

if [ x$mode == x"int8" ]; then
    quantize_args="--quantize W8BF16"
elif [ x$mode == x"bf16" ]; then
    quantize_args="--quantize BF16"
elif [ x$mode == x"fp16" ]; then
    quantize_args="--quantize F16"
elif [ x$mode == x"int4" ]; then
    quantize_args="--quantize W4BF16 --q_group_size 64"
else
    echo "Error, unknown quantize mode"
    exit 1
fi

timestamp=$(date "+%Y%m%d_%H%M%S")
out_model=${name}_${mode}_seq${seq_length}_1dev_${timestamp}.bmodel
onnx_folder="../../onnx"

if [ x$addr_mode == x"io_alone" ]; then
    addr_args="--addr_mode io_alone"
fi

embedding() {
    model_transform.py \
        --model_name embedding \
        --model_def $onnx_folder/embedding.onnx \
        --mlir embedding.mlir


    model_deploy.py \
        --mlir embedding.mlir \
        --quantize BF16 \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        --model embedding.bmodel
}

embedding_cache() {
    model_transform.py \
        --model_name embedding_cache \
        --model_def $onnx_folder/embedding.onnx \
        --input_shape [[1,1]] \
        --mlir embedding_cache.mlir


    model_deploy.py \
        --mlir embedding_cache.mlir \
        --quantize BF16 \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        --model embedding_cache.bmodel
}

process_block() {
    i="$1"
    model_transform.py \
        --model_name block_$i \
        --model_def $onnx_folder/block_$i.onnx \
        --mlir block_$i.mlir

    model_deploy.py \
        --mlir block_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def $onnx_folder/block_cache_$i.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $addr_args \
        --model block_cache_$i.bmodel
}

lm_head() {
    model_transform.py \
    --model_name lm_head \
    --model_def $onnx_folder/lm_head.pt \
    --input_shapes [[1,${hidden_size}]] \
    --mlir lm_head.mlir

    model_deploy.py \
        --mlir lm_head.mlir \
        $quantize_args \
        --quant_input \
        --chip bm1684x \
        --model lm_head.bmodel
}

greedy_head() {
    model_transform.py \
        --model_name greedy_head \
        --model_def $onnx_folder/greedy_head.onnx \
        --mlir greedy_head.mlir

    model_deploy.py \
        --mlir greedy_head.mlir \
        --chip bm1684x \
        --model greedy_head.bmodel
}

penalty_sample_head() {
    model_transform.py \
        --model_name penalty_sample_head \
        --model_def $onnx_folder/penalty_sample_head.onnx \
        --mlir penalty_sample_head.mlir

    model_deploy.py \
        --mlir penalty_sample_head.mlir \
        --chip bm1684x \
        --model penalty_sample_head.bmodel
}

vision_embedding() {
    model_transform.py \
        --model_name vision_embedding \
        --model_def $onnx_folder/vision_embedding.onnx \
        --mlir vision_embedding.mlir


    model_deploy.py \
        --mlir vision_embedding.mlir \
        --quantize BF16 \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        --model vision_embedding.bmodel
}

outdir=${folder}/$mode"_1dev"/embedding
mkdir -p $outdir
pushd $outdir
embedding
embedding_cache
models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '
rm -f *.npz
popd
echo $models

outdir=${folder}/$mode"_1dev"/lm_head
mkdir -p $outdir
pushd $outdir
lm_head
greedy_head
penalty_sample_head
models=${models}${outdir}'/lm_head.bmodel '$outdir'/greedy_head.bmodel '$outdir'/penalty_sample_head.bmodel '
rm -f *.npz
popd
echo $models

outdir=${folder}/$mode"_1dev"/block
mkdir -p $outdir
echo $outdir
pushd $outdir
# Process each block in parallel
for ((i=0; i<$num_layers; i++)); do
    # Check if block_$i.bmodel and block_cache_$i.bmodel exist
    if [ -f "block_$i.bmodel" ] && [ -f "block_cache_$i.bmodel" ]; then
        echo "block_$i.bmodel and block_cache_$i.bmodel already exist. Skipping..."
    else
        process_block $i &
        sleep 45
    fi
    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '
done
popd
echo $models

# Compile VIT model
outdir=${folder}/$mode"_1dev"/vit
mkdir -p $outdir
pushd $outdir
vision_embedding
models=${models}${outdir}'/vision_embedding.bmodel '
popd

model_tool --combine $models -o $out_model