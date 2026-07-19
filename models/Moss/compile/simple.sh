#!/bin/bash

# 默认从 0 到 33 运行
start=0
end=33

# 如果用户提供了参数，可以覆盖默认的起始和结束值
if [ $# -ge 1 ]; then
    start=$1
fi

if [ $# -ge 2 ]; then
    end=$2
fi

# 遍历数字范围并执行命令
for i in $(seq $start $end); do
    echo "Processing block_$i.onnx -> simple_block_$i.onnx"
    python -m onnxsim "block_$i.onnx" "simple_block_$i.onnx"
    python -m onnxsim "block_cache_$i.onnx" "simple_block_cache_$i.onnx"
    echo "Finished processing block_$i.onnx"
    echo "-----------------------------"
done

echo "All tasks completed!"
