#!/bin/bash
set -x

if [ ! -f "qwen3-4b-awq_w4bf16_seq512_bm1684x_1dev_20250514_161445.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-4b-awq_w4bf16_seq512_bm1684x_1dev_20250514_161445.bmodel
else
  echo "Bmodel Exists!"
fi

if ls ./python_demo/*cpython*.so 1> /dev/null 2>&1; then
  echo "cpython.so exists!"
else
  pushd python_demo
  rm -rf build && mkdir build && cd build
  cmake .. && make
  cp *cpython* ..
  popd
fi

echo $PWD
python3 python_demo/pipeline.py --model_path ./qwen3-4b-awq_w4bf16_seq512_bm1684x_1dev_20250514_161445.bmodel --config_path ./config --devid 0
