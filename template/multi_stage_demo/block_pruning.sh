#!/bin/bash
num_layers=28
process_block() {
    i=$1
    model_transform.py \
        --model_name block_$i \
        --model_def ./block_$i.onnx \
        --mlir block_$i.mlir \
        --pruning pruning_config_$i.json

    model_deploy.py \
        --mlir block_$i.mlir \
        --quantize W4F16 \
        --q_group_size 64 \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        --model block_$i.bmodel

    model_transform.py \
        --model_name block_cache_$i \
        --model_def ./block_cache_$i.onnx \
        --mlir block_cache_$i.mlir \
        --pruning pruning_config_$i.json

    model_deploy.py \
        --mlir block_cache_$i.mlir \
        --quantize W4F16 \
        --q_group_size 64 \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        --addr_mode io_alone \
        --model block_cache_$i.bmodel
}

python3 prun_onnx.py # gen test block onnx.
python3 fix_config.py
for ((i=0; i<$num_layers; i++)); do
    process_block $i &
        sleep 45
done
wait  # Wait for all background processes to finish
rm -f *.npz
popd


# test block_compare case
test_block_compare() {
    model_transform.py \
        --model_name block_model \
        --model_def pruned_model.onnx \
        --mlir block_model.mlir \
        --pruning pruning_config_0.json

    model_runner.py --input block_model_in_f32.npz --model block_model.mlir --output block_model_top_outputs.npz  

    model_deploy.py \
        --mlir block_model.mlir \
        --quantize W4F16 \
        --q_group_size 64 \
        --quant_input \
        --quant_output \
        --chip bm1684x \
        --test_input block_model_in_f32.npz \
        --test_reference block_model_ref_outputs.npz \
        --model mymatmul.bmodel
}
test_block_compare
