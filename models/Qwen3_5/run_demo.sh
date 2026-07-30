#!/bin/bash
set -x

if [ ! -f "qwen3.5-2b-int4-qwen3.5-2b-int4-autoround_w4bf16_seq8192_bm1684x_1dev_history_dynamic_20260729_163715.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3.5-2b-int4-qwen3.5-2b-int4-autoround_w4bf16_seq8192_bm1684x_1dev_history_dynamic_20260729_163715.bmodel
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
python3 python_demo/pipeline.py --model_path ./qwen3.5-2b-int4-qwen3.5-2b-int4-autoround_w4bf16_seq8192_bm1684x_1dev_history_dynamic_20260729_163715.bmodel --config_path ./config --devid 0
