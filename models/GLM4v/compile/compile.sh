#!/bin/bash
set -ex
# set -x
model_dir=$(dirname $(readlink -f "$0"))
pushd $model_dir

models=
mode="int4"
folder="tmp"
num_device=1
device_args=""
quantize_args="--quantize W4BF16 --q_group_size 64"
name="glm4v-9b"
num_layers=40
out_model=$name.bmodel
seq_length=2048

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

if [ "$name" = "glm4v-9b" ]; then
  num_layers=40
  hidden_size=4096
  image_size=1120
  echo "Compile GLM4v-9B"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mglm4v-9b\033[0m"
  exit 1
fi

if [ x$mode == x"int8" ]; then
    quantize_args="--quantize W8F16"
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
    out_model=$name'_'$mode'_'$num_device'dev.bmodel'
else
    out_model=$name'_'$mode'_1dev.bmodel'
fi

echo "数据格式 ${quantize_args}"
echo $models
# vision_transformer
outdir=${folder}/$mode"_"$num_device"dev"/vision_transformer
mkdir -p $outdir
pushd $outdir
#!/bin/bash
echo "当前目录是：$(pwd)"

model_transform.py \
    --model_name vision_transformer  \
    --model_def ../../vit/onnx/vision_transformers.pt \
    --input_shapes [[1,3,${image_size},${image_size}]] \
    --mlir vision_transformer.mlir

model_deploy.py \
    --mlir vision_transformer.mlir \
    $quantize_args \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    $device_args \
    --model vision_transformer.bmodel

rm *.npz *.onnx -f

models=$models' '$outdir'/vision_transformer.bmodel '

popd

outdir=${folder}/$mode"_"$num_device"dev"/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def ../../onnx/embedding.pt \
    --input_shapes [[1,${seq_length}]] \
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

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '

popd



outdir=tmp/$mode"_"$num_device"dev"/lm_head
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

rm -f *.npz

models=${models}${outdir}'/lm_head.bmodel '$outdir'/greedy_head.bmodel '$outdir'/penalty_sample_head.bmodel '
popd

echo $models

outdir=tmp/$mode"_"$num_device"dev"/block
mkdir -p $outdir

pushd $outdir
mkdir -p $outdir

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
        $device_args \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def ../../onnx/block_cache_$i.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $device_args \
        --addr_mode io_alone\
        --model block_cache_$i.bmodel
}

# Process each block in parallel
for ((i=0; i<$num_layers; i++)); do
    process_block $i
    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '
    sleep 45
done

wait  # Wait for all background processes to finish

rm -f *.npz
popd
echo $models

model_tool --combine $models -o $out_model
