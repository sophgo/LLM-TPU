#!/bin/bash
set -ex
models=""
mode="int4"
quantize_args=""
name="megrez"
seq_length=512

chip="bm1684x"
num_layers=32
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

if [ "$name" = "megrez" ]; then
  num_layers=32
  hidden_size=2560
  echo "Compile Megrez-3B-Instruct"
else
  echo -e "Error: Invalid name $name, the input name must be \033[31mmegrez\033[0m"
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

onnx_dir=$PWD/tmp/onnx
folder='tmp/'$name'_'$chip'_'$mode
out_model=$name'_'$chip'_'$mode'_seq'${seq_length}'.bmodel'

# Convert block
outdir=${folder}/block
mkdir -p $outdir
pushd $outdir

process_block()
{
    i=$1

    model_transform.py \
        --model_name block_$i \
        --model_def ${onnx_dir}/block_$i.onnx \
        --mlir block_$i.mlir

    model_deploy.py \
        --mlir block_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip ${chip} \
        $device_args \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def ${onnx_dir}/block_cache_$i.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip ${chip} \
        $device_args \
        --addr_mode io_alone \
        --model block_cache_$i.bmodel

    rm -f *.npz
}

# Process each block
for ((i=0; i<$num_layers; i++)); do
    process_block $i
    models="${models}${outdir}/block_${i}.bmodel ${outdir}/block_cache_${i}.bmodel "
done

popd
echo $models

# convert embedding
outdir=${folder}/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def ${onnx_dir}/embedding.pt \
    --input_shapes "[[1,$seq_length]]" \
    --input_types "int32" \
    --mlir embedding.mlir

model_deploy.py \
    --mlir embedding.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip ${chip} \
    $device_args \
    --model embedding.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def ${onnx_dir}/embedding.pt \
    --input_shapes "[[1,1]]" \
    --input_types "int32" \
    --mlir embedding_cache.mlir

model_deploy.py \
    --mlir embedding_cache.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip ${chip} \
    $device_args \
    --model embedding_cache.bmodel

rm -f *.npz

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '

popd
echo $models

# convert lm_head
outdir=${folder}/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def ${onnx_dir}/lm_head.pt \
    --input_shapes "[[1,${hidden_size}]]" \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    $quantize_args \
    --quant_input \
    --chip ${chip} \
    $device_args \
    --model lm_head.bmodel

rm -f *.npz

models=${models}${outdir}'/lm_head.bmodel '
popd

echo $models

model_tool --combine $models -o $out_model