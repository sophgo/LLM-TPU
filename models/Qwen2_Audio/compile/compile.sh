#!/bin/bash
set -ex

exe_dir=$(dirname $(readlink -f "$0"))
pushd $exe_dir

combined_outdir=../models/BM1684X/

if [ ! -d $combined_outdir ];
then
  mkdir -p $combined_outdir
else
  echo dir $combined_outdir exist
fi
echo $exe_dir
models=
folder=${exe_dir}"/../../models/onnx"
device_args=""
quantize_args="--quantize W8F16"
addr_args="--addr_mode io_alone"
name="qwen2-audio-7b"
num_layers=
out_model=$name.bmodel
seq_length=599
hidden_size=4096
mode="int8"
num_device=1
audio_seq_length=128

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
    --audio_seq_length)
        audio_seq_length="$2"
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

if [ "$name" = "qwen2-audio-7b" ]; then
  num_layers=32
  hidden_size=4096
  echo "Compile Qwen2-AUDIO-7B"
elif [ "$name" = "qwen2-audio-2b" ]; then
  num_layers=28
  hidden_size=1536
  echo "Compile Qwen2-Audio-2B"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mqwen2-vl-2b|qwen2-vl-7b\033[0m"
  exit 1
fi

if [ x$mode == x"int8" ]; then
    quantize_args="--quantize W8F16"
elif [ x$mode == x"bf16" ]; then
    quantize_args="--quantize BF16"
elif [ x$mode == x"fp16" ]; then
    quantize_args="--quantize F16"
elif [ x$mode == x"int4" ]; then
    quantize_args="--quantize W4BF16 --q_group_size 32"
else
    echo "Error, unknown quantize mode"
    exit 1
fi

timestamp=$(date "+%Y%m%d_%H%M%S")
out_model=${name}_${mode}_seq${seq_length}_1dev_${timestamp}.bmodel


if [ x$addr_mode == x"io_alone" ]; then
    addr_args="--addr_mode io_alone"
fi

outdir=${folder}/$mode"_1dev"/embedding
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name embedding \
    --model_def ${exe_dir}/../../llm/embedding.onnx \
    --input_shapes [[1,${seq_length}]] \
    --mlir embedding.mlir

model_deploy.py \
    --mlir embedding.mlir \
    --quant_input \
    --quantize W8F16 \
    --quant_output \
    --chip bm1684x \
    $device_args \
    $dyn_args \
    --model embedding.bmodel

model_transform.py \
    --model_name embedding_cache \
    --model_def ${exe_dir}/../../llm/embedding_cache.onnx \
    --input_shapes [[1,1]] \
    --mlir embedding_cache.mlir

model_deploy.py \
    --mlir embedding_cache.mlir \
    --quant_input \
    --quant_output \
    --chip bm1684x \
    --quantize W8F16 \
    $device_args \
    --model embedding_cache.bmodel

models=$models' '$outdir'/embedding.bmodel '$outdir'/embedding_cache.bmodel '

rm -f *.npz
popd
echo $models

outdir=${folder}/$mode"_1dev"/lm_head
mkdir -p $outdir
pushd $outdir

model_transform.py \
    --model_name lm_head \
    --model_def ${exe_dir}/../../llm/lm_head.pt \
    --input_shapes [[1,1,4096]] \
    --mlir lm_head.mlir

model_deploy.py \
    --mlir lm_head.mlir \
    --quantize W8F16 \
    --quant_input \
    --chip bm1684x \
    $device_args \
    --model lm_head.bmodel

models=$models' '$outdir'/lm_head.bmodel '
rm -f *.npz
popd
echo $models

outdir=${folder}/$mode"_1dev"/block
mkdir -p $outdir
echo $outdir
pushd $outdir

for ((i=0; i<$num_layers; i++)); do

    model_transform.py \
        --model_name block_$i \
        --model_def ${exe_dir}/../../llm/block_$i.onnx \
        --input_shapes [[1,599,4096],[1,599],[1,599]] \
        --mlir block_$i.mlir

    model_deploy.py \
        --mlir block_$i.mlir \
        --quantize W8F16 \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $device_args \
        $dyn_args \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def ${exe_dir}/../../llm/block_cache_$i.pt \
        --input_shapes [[1,1,4096],[1,1],[1,1,1,599],[1,32,599,128],[1,32,599,128]] \
        --mlir block_cache_$i.mlir

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        --quantize W8F16 \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        $device_args \
        $addr_args \
        $dyn_args \
        --model block_cache_$i.bmodel

    rm -f *.npz

    models=${models}${outdir}'/block_'$i'.bmodel '$outdir'/block_cache_'$i'.bmodel '

done
popd
echo $models

# Compile AUDIO model
outdir=${folder}/$mode"_1dev"/audio
mkdir -p $outdir
pushd $outdir
model_transform.py \
  --model_name audio \
  --model_def ${exe_dir}/../../llm/audio_ext_model.onnx \
  --input_shapes [[1,${audio_seq_length},3000],[1,1,1500,1500]] \
  --mlir audio.mlir

model_deploy.py \
  --mlir audio.mlir \
  --chip bm1684x \
    --quantize W8F16 \
  --model audio.bmodel

popd
models=$models' '$outdir'/audio.bmodel '

# Compile projector model
outdir=${folder}/$mode"_1dev"/projector
mkdir -p $outdir
pushd $outdir
model_transform.py \
  --model_name projector \
  --model_def ${exe_dir}/../../llm/multi_modal_projector.onnx \
  --input_shapes [[1,750,1280]] \
  --mlir projector.mlir

model_deploy.py \
  --mlir projector.mlir \
  --chip bm1684x \
  --quantize W8F16 \
  --model projector.bmodel

popd
models=$models' '$outdir'/projector.bmodel '


# Compile greed model
outdir=${folder}/$mode"_1dev"/greed
mkdir -p $outdir
pushd $outdir
model_transform.py \
  --model_name greed \
  --model_def ${exe_dir}/../../llm/greed.pt \
  --input_shapes [[1,1,115630]] \
  --mlir greed.mlir

model_deploy.py \
  --mlir greed.mlir \
  --chip bm1684x \
  --quantize W8F16 \
  --model greed.bmodel

popd
models=$models' '$outdir'/greed.bmodel '

model_tool --combine $models  -o $out_model
chmod 666 $out_model
mv $out_model $combined_outdir