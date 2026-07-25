#!/bin/bash
#./compile.sh --name qwen2.5-vl-3b --seq_length 2048
set -ex
models=
quantize_args=""
name=""
num_layers=
hidden_size=
mode="w4bf16"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
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

if [ "$name" = "drivemm" ]; then
  num_layers=32
  hidden_size=4096
  echo "Compile DriveMM"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mdrivemm\033[0m"
  exit 1
fi

quantize_args="--quantize W4BF16 --q_group_size 128"
half_quantize_args="--quantize BF16"

timestamp=$(date "+%Y%m%d_%H%M%S")
out_model=${name}_${mode}_seq${seq_length}_1dev_${timestamp}.bmodel
TASK_FILE="task.txt"


MODEL_DIR=${DIR}/tmp/onnx
COMPILE_DIR=${DIR}/tmp/$mode"_1dev"
TASK_FILE=${COMPILE_DIR}/${TASK_FILE}


embedding() {
    echo \
    model_convert.py \
        --model_name embedding \
        --model_def ${MODEL_DIR}/embedding.pt \
        --input_shapes [[1,${seq_length}]] \
        --input_types "int32" \
        ${quantize_args} \
        --quant_input \
        --quant_output \
        --debug \
        --chip bm1684x \
        --model embedding.bmodel \
        >> ${TASK_FILE}

    models=${models}${COMPILE_DIR}'/embedding.bmodel '

    echo \
    model_convert.py \
        --model_name embedding_cache \
        --model_def ${MODEL_DIR}/embedding.pt \
        --input_shapes [[1,1]] \
        --input_types "int32" \
        ${quantize_args} \
        --quant_input \
        --quant_output \
        --debug \
        --chip bm1684x \
        --model embedding_cache.bmodel \
        >> ${TASK_FILE}

    models=${models}${COMPILE_DIR}'/embedding_cache.bmodel '

}

lm_head() {
    echo \
    model_convert.py \
        --model_name lm_head \
        --model_def ${MODEL_DIR}/lm_head.pt \
        --input_shapes [[1,${hidden_size}]] \
        ${half_quantize_args} \
        --quant_input \
        --debug \
        --chip bm1684x \
        --model lm_head.bmodel \
        >> ${TASK_FILE}

    models=${models}${COMPILE_DIR}'/lm_head.bmodel '

    echo \
    model_convert.py \
        --model_name greedy_head \
        --model_def ${MODEL_DIR}/greedy_head.onnx \
        ${half_quantize_args} \
        --debug \
        --chip bm1684x \
        --model greedy_head.bmodel \
        >> ${TASK_FILE}

    models=${models}${COMPILE_DIR}'/greedy_head.bmodel '

    echo \
    model_convert.py \
        --model_name penalty_sample_head \
        --model_def ${MODEL_DIR}/penalty_sample_head.onnx \
        ${half_quantize_args} \
        --chip bm1684x \
        --model penalty_sample_head.bmodel \
        >> ${TASK_FILE}

    models=${models}${COMPILE_DIR}'/penalty_sample_head.bmodel '
}

block() {
    for ((i=0; i<$num_layers; i++)); do
        echo \
        model_convert.py \
            --model_name block_$i \
            --model_def ${MODEL_DIR}/block_$i.onnx \
            ${quantize_args} \
            --quant_input \
            --quant_output \
            --chip bm1684x \
            --model block_$i.bmodel \
            >> ${TASK_FILE}

        echo \
        model_convert.py \
            --model_name block_cache_$i \
            --model_def ${MODEL_DIR}/block_cache_$i.onnx \
            ${quantize_args} \
            --quant_input \
            --quant_output \
            --chip bm1684x \
            --addr_mode io_alone \
            --model block_cache_$i.bmodel \
            >> ${TASK_FILE}
        
        models=${models}${COMPILE_DIR}'/block_'$i'.bmodel '${COMPILE_DIR}'/block_cache_'$i'.bmodel '
    done
}

vision_transformer() {
    echo \
    model_convert.py \
        --model_name vit \
        --model_def ${MODEL_DIR}/image_encoder.onnx \
        --do_onnx_sim True \
        ${half_quantize_args} \
        --quant_output \
        --chip bm1684x \
        --model vit.bmodel \
        >> ${TASK_FILE}

    models=${models}${COMPILE_DIR}'/vit.bmodel '
}

mkdir -p $COMPILE_DIR
echo $COMPILE_DIR
: > ${TASK_FILE}
pushd $COMPILE_DIR
vision_transformer
block
embedding
lm_head

num_cores=$(grep -c ^processor /proc/cpuinfo)
parallel -j ${num_cores} --progress --joblog ${TASK_FILE}.log < ${TASK_FILE}
[[ $? -ne 0 ]] && { echo "Error: model convert failed"; exit 1; }

rm -f *.npz *.onnx
popd

model_tool --combine $models -o $out_model
echo "Success: gen model $out_model"


