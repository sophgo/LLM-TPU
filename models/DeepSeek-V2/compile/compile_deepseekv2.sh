#!/bin/bash

NUM_LAYERS=2
NUM_EXPERTS=160
out_model="combined_model.bmodel"  # 定义最终合并模型名称
models=()  # 初始化模型路径数组

pushd tmp/bmodel

# 处理设备0的层0
for device_id in 0; do
    # 转换第0层模型
    for layer_id in 0; do
        echo "Converting layer $layer_id on device $device_id"
        
        # Attention 模块
        model_convert.py --model_name "attention_${layer_id}" \
            --model_def "../onnx/attention_layer${layer_id}_dev${device_id}.onnx" \
            --quantize w4bf16 --quant_input --quant_output --chip bm1684x \
            --model "attention_${layer_id}.bmodel" --do_onnx_sim True
        models+=("attention_${layer_id}.bmodel")

        # Attention Cache 模块
        model_convert.py --model_name "attention_cache_${layer_id}" \
            --model_def "../onnx/attention_cache_layer${layer_id}_dev${device_id}.onnx" \
            --quantize w4bf16 --quant_input --quant_output --chip bm1684x \
            --model "attention_cache_${layer_id}.bmodel" --do_onnx_sim True
        models+=("attention_cache_${layer_id}.bmodel")

        # MLP 模块
        model_convert.py --model_name "mlp_${layer_id}" \
            --model_def "../onnx/mlp_layer${layer_id}_dev${device_id}.onnx" \
            --quantize w4bf16 --quant_input --quant_output --chip bm1684x \
            --model "mlp_${layer_id}.bmodel" --do_onnx_sim True
        models+=("mlp_${layer_id}.bmodel")

        # MLP Cache 模块
        model_convert.py --model_name "mlp_cache_${layer_id}" \
            --model_def "../onnx/mlp_cache_layer${layer_id}_dev${device_id}.onnx" \
            --quantize w4bf16 --quant_input --quant_output --chip bm1684x \
            --model "mlp_cache_${layer_id}.bmodel" --do_onnx_sim True
        models+=("mlp_cache_${layer_id}.bmodel")
    done

    # 转换1到NUM_LAYERS-1层模型
    for (( layer_id=1; layer_id<NUM_LAYERS; layer_id++ )); do
        echo "Converting layer $layer_id on device $device_id"
        
        # Attention 模块
        model_convert.py --model_name "attention_${layer_id}" \
            --model_def "../onnx/attention_layer${layer_id}_dev${device_id}.onnx" \
            --quantize w4bf16 --quant_input --quant_output --chip bm1684x \
            --model "attention_${layer_id}.bmodel" --do_onnx_sim True
        models+=("attention_${layer_id}.bmodel")

        # Attention Cache 模块
        model_convert.py --model_name "attention_cache_${layer_id}" \
            --model_def "../onnx/attention_cache_layer${layer_id}_dev${device_id}.onnx" \
            --quantize w4bf16 --quant_input --quant_output --chip bm1684x \
            --model "attention_cache_${layer_id}.bmodel" --do_onnx_sim True
        models+=("attention_cache_${layer_id}.bmodel")

        # Shared MoE 模块
        model_convert.py --model_name "shared_moe_${layer_id}" \
            --model_def "../onnx/shared_moe_layer${layer_id}_dev${device_id}.onnx" \
            --quantize w4bf16 --quant_input --quant_output --chip bm1684x \
            --model "shared_moe_${layer_id}.bmodel" --do_onnx_sim True
        models+=("shared_moe_${layer_id}.bmodel")

        # Shared MoE Cache 模块
        model_convert.py --model_name "shared_moe_cache_${layer_id}" \
            --model_def "../onnx/shared_moe_cache_layer${layer_id}_dev${device_id}.onnx" \
            --quantize w4bf16 --quant_input --quant_output --chip bm1684x \
            --model "shared_moe_cache_${layer_id}.bmodel" --do_onnx_sim True
        models+=("shared_moe_cache_${layer_id}.bmodel")

        # MoE 模块
        model_convert.py --model_name "moe_${layer_id}" \
            --model_def "../onnx/moe_layer${layer_id}_dev${device_id}.onnx" \
            --quantize w4bf16 --quant_input --quant_output --chip bm1684x \
            --model "moe_${layer_id}.bmodel" --do_onnx_sim True
        models+=("moe_${layer_id}.bmodel")

        # MoE Cache 模块（每个专家）
        for expert_id in $(seq 0 $((NUM_EXPERTS - 1))); do
            model_convert.py --model_name "moe_cache_${layer_id}_${expert_id}" \
                --model_def "../onnx/moe_cache_expert${expert_id}_layer${layer_id}_dev${device_id}.onnx" \
                --quantize w4bf16 --quant_input --quant_output --chip bm1684x \
                --model "moe_cache_${layer_id}_${expert_id}.bmodel" --do_onnx_sim True
            models+=("moe_cache_${layer_id}_${expert_id}.bmodel")
        done
    done
done

# 合并所有模型
echo "Combining all models..."
model_tool --combine "${models[@]}" -o "$out_model"

popd