#!/bin/bash
set -ex
models=
mode="f16"
folder="tmp"
num_device=1
device_args=""
addr_args=""
quantize_args="--quantize F32"
name=""
num_layers=
hidden_size=
out_model=$name.bmodel

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

if [ "$name" = "rwkv6-1b6" ]; then
 num_layers=23
 hidden_size=2048
 state_size_1=1584
 state_size_2=$hidden_size
 echo "Compile RWKV6-1B6"
elif [ "$name" = "rwkv6-3b" ]; then
 num_layers=31
 hidden_size=2560
 state_size_1=2624
 state_size_2=$hidden_size
 echo "Compile RWKV6-3B"
else
 >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mrwkv6-1b5|rwkv6-3b\033[0m"
 exit 1
fi

if [ x$mode == x"int8" ]; then
    quantize_args="--quantize W8BF16"
elif [ x$mode == x"bf16" ]; then
    quantize_args="--quantize BF16"
elif [ x$mode == x"int4" ]; then
    quantize_args="--quantize W4BF16 --q_group_size 64"
else
    echo "Error, unknown quantize mode (Now only support INT4/INT8/BF16)"
    exit 1
fi

if [ x$num_device != x1 ]; then
    device_args="--num_device $num_device"
    out_model=$name'_'$mode'_'$num_device'dev.bmodel'
else
    out_model=$name'_'$mode'_1dev.bmodel'
fi

# Make MLIR
# convert emb
outdir=${folder}/embedding
mkdir -p $outdir
pushd $outdir
model_transform.py \
    --model_name embedding \
    --model_def ../onnx/embedding.onnx \
    --mlir embedding.mlir
model_deploy.py \
    --mlir embedding.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    $device_args \
    --model embedding.bmodel
rm *.npz *.onnx -f
models=$models' '$outdir'/embedding.bmodel '
popd
echo $models


# convert lm_head
outdir=${folder}/$mode"_"$num_device"dev"/lm_head
mkdir -p $outdir
pushd $outdir
model_transform.py \
    --model_name lm_head \
    --model_def ../../onnx/lm_head.onnx \
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

rm *.npz *.onnx -f
models=${models}${outdir}'/lm_head.bmodel '$outdir'/greedy_head.bmodel '$outdir'/penalty_sample_head.bmodel '
popd
echo $models

# convert blocks
outdir=${folder}/$mode"_"$num_device"dev"/block
mkdir -p $outdir
pushd $outdir

for ((i=0; i<=$num_layers; i++)); do
    model_transform.py \
        --model_name block_$i \
        --model_def ../../onnx/block_$i.onnx \
        --input_shapes [[1,${hidden_size}],[1,${state_size_1},${state_size_2}]] \
        --mlir block_$i.mlir


    model_deploy.py \
        --mlir block_$i.mlir \
        $quantize_args \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $device_args \
        --model block_$i.bmodel
    rm *.npz *.onnx -f
    models=${models}${outdir}'/block_'$i'.bmodel '

done
popd
echo $models

model_tool --combine $models -o $out_model
