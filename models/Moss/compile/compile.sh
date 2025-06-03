#!/bin/bash
set -ex

models=
models_2core=
mode="int4"
folder="/workspace/scripts/models_7"
target_dir="BM1684X"
num_device=1
mode_args=""
device_args=""
#quantize_args="--quantize W8BF16"
quantize_args="--quantize W4BF16 --q_group_size 64"
addr_args=""
dyn_args=""
name="moss"
num_layers=34
# num_layers=1
out_model=$name.bmodel
out_model_2core=$name_2core.bmodel
seq_length=512
hidden_size=6144
dynamic=0
batch=1
target=bm1684x
chip_args="bm1684x"

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    --mode)
        mode="$2"
        shift 2
        ;;
    --target)
        target="$2"
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


#if [ x$mode == x"int8" ]; then
#    quantize_args="--quantize W8BF16"
#elif [ x$mode == x"bf16" ]; then
#    quantize_args="--quantize BF16"
#elif [ x$mode == x"f16" ]; then
#    quantize_args="--quantize F16"
#elif [ x$mode == x"int4" ]; then
#    quantize_args="--quantize W4BF16 --q_group_size 64"
#else
#    echo "Error, unknown quantize mode"
#    exit 1
#fi

if [ "$target" != "bm1684x" ] && [ "$target" != "bm1688" ] && [ "$target" != "cv186x" ];
then
    >&2 echo -e "Error: Invalid target $target, the input target must be \033[31mbm1684x|bm1688|cv186x\033[0m"
    exit 1
fi

if [ x$num_device != x1 ]; then
    device_args="--num_device $num_device"
    out_model=${name}_${mode}_seq${seq_length}_${num_device}dev.bmodel
else
    out_model=${name}_${mode}_seq${seq_length}_1dev.bmodel
fi



if [ x$addr_mode == x"io_alone" ]; then
    addr_args="--addr_mode io_alone"
fi

if [ x$dynamic == x1 ]; then
    if [ "$target" != "bm1684x" ]; then
        echo "dynamic is not supported on $target"
        exit
    fi
    dyn_args="--dynamic"
    out_model=${name}_${mode}_seq${seq_length}_${num_device}dev_dyn.bmodel
fi

process_block() {
    i=$1
    model_transform.py \
    --model_name block_$i \
    --model_def /workspace/scripts/onnx2/simple_block_$i.onnx \
    --mlir block_$i.mlir

    model_deploy.py \
        --mlir block_$i.mlir \
        ${quantize_args} \
        --quant_input \
        --quant_output \
        --chip $chip_args \
        $device_args \
        $dyn_args \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def /workspace/scripts/onnx2/simple_block_cache_$i.onnx \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        ${quantize_args} \
        --quant_input \
        --quant_output \
        --chip $chip_args \
        $device_args \
        $dyn_args \
        $addr_args \
        --model block_cache_$i.bmodel

    
}

outdir=${folder}/$mode"_"$num_device"dev"/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def /workspace/scripts/onnx2/embedding.pt \
    --input_shapes [[1,${seq_length}]] \
    --input_types "int32" \
    --mlir embedding.mlir

model_deploy.py \
    --mlir embedding.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip $chip_args \
    $device_args \
    $dyn_args \
    --model embedding.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def /workspace/scripts/onnx2/embedding.pt \
    --input_shapes [[1,1]] \
    --input_types "int32" \
    --mlir embedding_cache.mlir

model_deploy.py \
    --mlir embedding_cache.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip $chip_args \
    $device_args \
    --model embedding_cache.bmodel

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '

if [ "$target" == "bm1688" ];
then
model_deploy.py \
    --mlir embedding.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip $chip_args --num_core 2 \
    $device_args \
    $dyn_args \
    --model embedding_2core.bmodel

model_deploy.py \
    --mlir embedding_cache.mlir \
    --quantize BF16 \
    --quant_input \
    --quant_output \
    --chip $chip_args --num_core 2 \
    $device_args \
    --model embedding_cache_2core.bmodel

models_2core=$models_2core' '$outdir'/embedding_2core.bmodel '$outdir'/embedding_cache_2core.bmodel '
fi

rm -f *.npz
popd

echo $models

outdir=${folder}/$mode"_"$num_device"dev"/lm_head
mkdir -p $outdir
pushd $outdir

if [[ $num_device -gt 1 ]]; then
    model_transform.py \
        --model_name lm_head \
        --model_def /workspace/scripts/onnx2/lm_head_with_topk.pt \
        --input_shapes [[1,1,${hidden_size}]] \
        --mlir lm_head.mlir

    model_deploy.py \
        --mlir lm_head.mlir \
        ${quantize_args} \
        --quant_input \
        --chip bm1684x \
        $device_args \
        --model lm_head.bmodel

    models=${models}${outdir}'/lm_head.bmodel '
else
    model_transform.py \
        --model_name lm_head \
        --model_def /workspace/scripts/onnx2/lm_head.pt \
        --input_shapes [[1,${hidden_size}]] \
        --mlir lm_head.mlir

    model_deploy.py \
        --mlir lm_head.mlir \
        $quantize_args \
        --quant_input \
        --chip $chip_args \
        $device_args \
        --model lm_head.bmodel


    model_transform.py \
        --model_name greedy_head \
        --model_def /workspace/scripts/onnx2/greedy_head.onnx \
        --mlir greedy_head.mlir

    model_deploy.py \
        --mlir greedy_head.mlir \
        --chip $chip_args \
        --model greedy_head.bmodel


    model_transform.py \
        --model_name penalty_sample_head \
        --model_def /workspace/scripts/onnx2/penalty_sample_head.onnx \
        --mlir penalty_sample_head.mlir

    model_deploy.py \
        --mlir penalty_sample_head.mlir \
        --chip $chip_args \
        --model penalty_sample_head.bmodel

    if [ "$target" == "bm1688" ];
    then
    model_deploy.py \
        --mlir lm_head.mlir \
        $quantize_args \
        --quant_input \
        --chip $chip_args --num_core 2 \
        $device_args \
        --model lm_head_2core.bmodel
    models_2core=${models_2core}' '${outdir}'/lm_head_2core.bmodel '
    fi

    if [ "$target" == "bm1688" ] || [ "$target" == "cv186x" ]; then
        models=${models}' '${outdir}'/lm_head.bmodel '
    else
        models=${models}${outdir}'/lm_head.bmodel '$outdir'/greedy_head.bmodel '$outdir'/penalty_sample_head.bmodel '
    fi
fi

rm -f *.npz
popd
echo $models

outdir=${folder}/$mode"_"$num_device"dev"/block
mkdir -p $outdir
pushd $outdir
for ((i=0; i<$num_layers; i++)); do
    process_block $i
    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '
    if [ "$target" == "bm1688" ]; then
        models_2core=${models_2core}${outdir}'/block_'$i'_2core.bmodel '$outdir'/block_cache_'$i'_2core.bmodel '
    fi
done

wait
rm -f *.npz
popd
echo $models

outdir=$folder/$target_dir/
if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

model_tool --combine $models -o ${outdir}${out_model}

