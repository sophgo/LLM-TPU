#!/bin/bash
set -ex
models=
mode="int8"
folder="tmp"
num_device=1
device_args=""
quantize_args="--quantize W8BF16"
addr_args=""
dyn_args=""
future_update_args=""
name=""
num_layers=
out_model=""
prefill_length=
hidden_size=
dynamic=0
max_rank_num=0

echo "必须把seq_length长的放在前面，这一点尤其要注意"

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
    --prefill_length_list)
        prefill_length_list="$2"
        shift 2
        ;;
    --seq_length_list)
        seq_length_list="$2"
        shift 2
        ;;
    --dynamic)
        dynamic="$2"
        shift 2
        ;;
    --generation_mode)
        generation_mode="$2"
        shift 2
        ;;
    --embedding_mode)
        embedding_mode="$2"
        shift 2
        ;;
    --max_rank_num)
        max_rank_num="$2"
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

if [[ -z "$prefill_length_list" ]]; then
    echo "Error: --prefill_length_list is required." >&2
    exit 1
fi

if [[ -z "$seq_length_list" ]]; then
    echo "Error: --seq_length_list is required." >&2
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

# Split lists into arrays
IFS=',' read -r -a prefill_lengths <<< "$prefill_length_list"
IFS=',' read -r -a seq_lengths <<< "$seq_length_list"


# Loop to process different models
for index in "${!prefill_lengths[@]}"; do

    prefill_length=${prefill_lengths[$index]}
    seq_length=${seq_lengths[$index]}
    folder_suffix="prefill${prefill_length}_seq${seq_length}"
    folder="tmp_${folder_suffix}"

    folder="tmp_prefill${prefill_length}_seq${seq_length}"

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

    cur_model=${name}_${mode}_${folder_suffix}
    if [ x$num_device != x1 ]; then
        device_args="--num_device $num_device"
        cur_model=${cur_model}_${num_device}dev
    else
        cur_model=${cur_model}_1dev
    fi

    if [ x$dynamic == x1 ]; then
        dyn_args="--dynamic"
        cur_model=${cur_model}_dyn
    fi

    if [ x$max_rank_num != x0 ]; then
        future_update_args="--future_update_rank ${max_rank_num} --disable_layer_group"
        cur_model=${cur_model}_rank${max_rank_num}
    else
        >&2 echo -e "Error: the max_rank_num is equal to zero"
        exit 1
    fi

    # out_model=${out_model}${cur_model}_

    if [ x$addr_mode == x"io_alone" ]; then
        addr_args="--addr_mode io_alone"
    fi

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
            $future_update_args \
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
            $device_args \
            $addr_args \
            $future_update_args \
            --model block_cache_$i.bmodel
    }

    # Process each block in parallel
    for ((i=0; i<$num_layers; i++)); do
        process_block $i &
        models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '
        sleep 45
    done

    wait  # Wait for all background processes to finish

    rm -f *.npz
    popd
    echo $models


    if [ x$embedding_mode == x"default" ]; then
        outdir=${folder}/$mode"_"$num_device"dev"/embedding
        mkdir -p $outdir
        pushd $outdir

        model_transform.py \
            --model_name embedding \
            --model_def ../../onnx/embedding.pt \
            --input_shapes [[1,${prefill_length}]] \
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

        models=${models}$outdir'/embedding.bmodel '

        if [ x$index == x"0" ]; then
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
            models=${models}$outdir'/embedding_cache.bmodel '
        fi

        rm -f *.npz
        popd
        echo $models
    fi

    if [ x$max_embedding_rank_num != x"0" ]; then
        outdir=${folder}/$mode"_"$num_device"dev"/embedding
        mkdir -p $outdir
        pushd $outdir

        model_transform.py \
            --model_name lora_embedding \
            --model_def ../../onnx/lora_embedding.onnx \
            --input_shapes [[1,${prefill_length}],[1,${prefill_length},${hidden_size}]] \
            --input_types "int32" \
            --mlir lora_embedding.mlir

        model_deploy.py \
            --mlir lora_embedding.mlir \
            --quantize BF16 \
            --quant_input \
            --quant_output \
            --chip bm1684x \
            $device_args \
            --model lora_embedding.bmodel
        models=${models}$outdir'/lora_embedding.bmodel '

        if [ x$index == x"0" ]; then
            model_transform.py \
                --model_name lora_embedding_cache \
                --model_def ../../onnx/lora_embedding.onnx \
                --input_shapes [[1,1],[1,1,${hidden_size}]] \
                --mlir lora_embedding_cache.mlir

            model_deploy.py \
                --mlir lora_embedding_cache.mlir \
                --quantize BF16 \
                --quant_input \
                --quant_output \
                --chip bm1684x \
                $device_args \
                --model lora_embedding_cache.bmodel
            models=${models}$outdir'/lora_embedding_cache.bmodel '
        fi

        rm -f *.npz
        popd
        echo $models
    fi

    outdir=${folder}/$mode"_"$num_device"dev"/lm_head
    mkdir -p $outdir
    pushd $outdir

    if [ x$index == x"0" ]; then
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

        models=${models}${outdir}'/lm_head.bmodel '$outdir'/greedy_head.bmodel '
    fi

    model_transform.py \
        --model_name penalty_sample_head \
        --model_def ../../onnx/penalty_sample_head.onnx \
        --mlir penalty_sample_head.mlir

    model_deploy.py \
        --mlir penalty_sample_head.mlir \
        --chip bm1684x \
        --model penalty_sample_head.bmodel

    rm *.npz *.onnx -f

    models=${models}$outdir'/penalty_sample_head.bmodel '

    popd
    echo $models
    rm -rf /root/.cache/tpu-mlir
done

model_tool --combine $models -o ${name}.bmodel