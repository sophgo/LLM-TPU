#!/bin/bash
# ./compile.sh --mode int4 --seq_length 4096
set -ex
models=
mode="bf16"
folder="tmp"
quantize_args="--quantize BF16"
name="qwen2.5-3b"
num_layers=
hidden_size=
seq_length=
out_model=$name.bmodel

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

if [ "$name" = "qwen2.5-14b" ]; then
    num_layers=48
    hidden_size=5120
    echo "Compile Qwen2.5-14B"
elif [ "$name" = "qwen2.5-7b" ]; then
    num_layers=28
    hidden_size=3584
    echo "Compile Qwen2.5-7B"
elif [ "$name" = "qwen2.5-3b" ]; then
    num_layers=36
    hidden_size=2048
    echo "Compile Qwen2.5-3B"
elif [ "$name" = "qwen2.5-1.5b" ]; then
    num_layers=28
    hidden_size=1536
    echo "Compile Qwen2.5-1.5B"
else
    echo >&2 -e "Error: Invalid name $name, the input name must be \033[31mqwen2.5-14b|qwen2.5-7b|qwen2.5-3b|qwen2.5-1.5b\033[0m"
    exit 1
fi

if [[ -z "$seq_length" ]]; then
    echo "Error: --seq_length is required." >&2
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

out_model=$name'_'$mode'_seq'$seq_length'.bmodel'

outdir=${folder}/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def ../onnx/embedding.pt \
    --input_shapes "[[1,$seq_length]]" \
    --input_types "int32" \
    --mlir embedding.mlir

model_deploy.py \
    --mlir embedding.mlir \
    --quantize BF16 \
    --quant_output \
    --chip bm1684x \
    --model embedding.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def ../onnx/embedding.pt \
    --input_shapes "[[1,1]]" \
    --input_types "int32" \
    --mlir embedding_cache.mlir

model_deploy.py \
    --mlir embedding_cache.mlir \
    --quantize BF16 \
    --quant_output \
    --chip bm1684x \
    --model embedding_cache.bmodel

rm *.npz

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '

popd

echo $models

outdir=${folder}/$mode/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def ../../onnx/lm_head.pt \
    --input_shapes "[[1,${hidden_size}]]" \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    $quantize_args \
    --quant_input \
    --chip bm1684x \
    --model lm_head.bmodel

rm *.npz
models=${models}${outdir}'/lm_head.bmodel'

popd
echo $models

outdir=${folder}/$mode/block
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
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def ../../onnx/block_cache_${i}.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        --addr_mode io_alone \
        --model block_cache_$i.bmodel
}
# Process each block in parallel
for ((i = 0; i < $num_layers; i++)); do
    process_block $i &
    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '
    sleep 45
done

wait # Wait for all background processes to finish

rm -f *.npz
popd
echo $models

model_tool --combine $models -o $out_model
