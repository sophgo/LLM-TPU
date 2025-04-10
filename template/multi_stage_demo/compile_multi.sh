#!/bin/bash
set -ex
models=
folder=""
quantize_args=""
dyn_args=""
name=""
num_layers=
out_model=$name.bmodel
hidden_size=
mode="w4bf16"

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
    --folder)
        folder="$2"
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

if [ "$name" = "qwen2-vl-7b" ]; then
  num_layers=28
  hidden_size=3584
  echo "Compile Qwen2-VL-7B"
elif [ "$name" = "qwen2-vl-2b" ]; then
  num_layers=28
  hidden_size=1536
  echo "Compile Qwen2-VL-2B"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mqwen2-vl-2b|qwen2-vl-7b\033[0m"
  exit 1
fi

if [ x$mode == x"w8bf16" ]; then
    quantize_args="--quantize W8BF16"
elif [ x$mode == x"w4bf16" ]; then
    quantize_args="--quantize W4BF16 --q_group_size 64"
elif [ x$mode == x"w8f16" ]; then
    quantize_args="--quantize W8F16 --q_group_size 64"
elif [ x$mode == x"w4f16" ]; then
    quantize_args="--quantize W4F16 --q_group_size 64"
elif [ x$mode == x"bf16" ]; then
    quantize_args="--quantize BF16"
elif [ x$mode == x"f16" ]; then
    quantize_args="--quantize F16"
else
    echo "Error, unknown quantize mode"
    exit 1
fi

if [[ "$mode" == "w8bf16" || "$mode" == "w4bf16" || "$mode" == "bf16" ]]; then
    half_quantize_args="--quantize BF16"
elif [[ "$mode" == "w8f16" || "$mode" == "w4f16" || "$mode" == "f16" ]]; then
    half_quantize_args="--quantize F16"
fi

timestamp=$(date "+%Y%m%d_%H%M%S")
out_model=${name}_${mode}_seq${seq_length}_1dev_${timestamp}.bmodel

embedding() {
    model_transform.py \
        --model_name embedding \
        --model_def ../../onnx/embedding.pt \
        --input_shapes [[1,${seq_length}]] \
        --input_types "int32" \
        --mlir embedding.mlir

    model_deploy.py \
        --mlir embedding.mlir \
        ${quantize_args} \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $dyn_args \
        --model embedding.bmodel

    rm -f *.npz
    models=${models}${outdir}'/embedding.bmodel '
}

embedding_cache() {
    model_transform.py \
        --model_name embedding_cache \
        --model_def ../../onnx/embedding.pt \
        --input_shapes [[1,1]] \
        --input_types "int32" \
        --mlir embedding_cache.mlir

    model_deploy.py \
        --mlir embedding_cache.mlir \
        ${quantize_args} \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        --model embedding_cache.bmodel

    rm -f *.npz
    models=${models}${outdir}'/embedding_cache.bmodel '
}

lm_head() {
    model_transform.py \
        --model_name lm_head \
        --model_def ../../onnx/lm_head.pt \
        --input_shapes [[1,${hidden_size}]] \
        --mlir lm_head.mlir

    model_deploy.py \
        --mlir lm_head.mlir \
        ${quantize_args} \
        --quant_input \
        --chip bm1684x \
        --model lm_head.bmodel

    rm -f *.npz
    # models=${models}${outdir}'/lm_head.bmodel '
}

greedy_head() {
    model_transform.py \
        --model_name greedy_head \
        --model_def ../../onnx/greedy_head.onnx \
        --mlir greedy_head.mlir

    model_deploy.py \
        --mlir greedy_head.mlir \
        --chip bm1684x \
        --model greedy_head.bmodel

    rm -f *.npz
    # models=${models}${outdir}'/greedy_head.bmodel '
}

penalty_sample_head() {
    model_transform.py \
        --model_name penalty_sample_head \
        --model_def ../../onnx/penalty_sample_head.onnx \
        --mlir penalty_sample_head.mlir

    model_deploy.py \
        --mlir penalty_sample_head.mlir \
        --chip bm1684x \
        --model penalty_sample_head.bmodel

    rm -f *.npz
    models=${models}${outdir}'/penalty_sample_head.bmodel '
}

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
        --chip bm1684x \
        $dyn_args \
        --model block_$i.bmodel

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
        --addr_mode io_alone \
        --model block_cache_$i.bmodel
}

vision_transformer() {
    model_transform.py \
        --model_name vit \
        --model_def ../../onnx/vit/vit.onnx \
        --mlir vit.mlir

    model_deploy.py \
        --mlir vit.mlir \
        ${half_quantize_args} \
        --quant_input \
        --quant_input_list 3 \
        --quant_output \
        --high_precision \
        --chip bm1684x \
        --model vit.bmodel

    rm -f *.npz
    # models=${models}${outdir}'/vit.bmodel '
}

# outdir=${folder}/$mode"_1dev"/embedding
# mkdir -p $outdir
# pushd $outdir
# embedding
# embedding_cache
# popd

outdir=${folder}/$mode"_1dev"/lm_head
mkdir -p $outdir
pushd $outdir
lm_head
greedy_head
penalty_sample_head
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
    models=${models}${outdir}'/block_'$i'.bmodel '${outdir}'/block_cache_'$i'.bmodel '
done
wait  # Wait for all background processes to finish
rm -f *.npz
popd
echo $models

# Compile VIT model
outdir=${folder}/$mode"_1dev"/vit


mkdir -p $outdir
pushd $outdir
vision_transformer
popd

model_tool --combine $models -o $outdir/../$out_model
model_tool --info $outdir/../$out_model > $outdir/../model.log

# shape 按从小到大做combine
# 注意：只有 block 、 block_cache 、 penalty_sample_head 三个模块有多stage，因此 lm_head 、 vit 、 greedy_head 只能combine一次，否则会报错！
