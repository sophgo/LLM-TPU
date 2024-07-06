#!/bin/bash
set -ex

# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/qwen-7b_int4_1dev_seq512.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen-7b_int4_1dev_seq512.bmodel
  mv qwen-7b_int4_1dev_seq512.bmodel ../../bmodels
else
  echo "Bmodel Exists!"
fi

if [ ! -f "./python_demo/*cpython*" ]; then
  cd python_demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j
  cp *cpython* ..
  cd ../..
else
  echo "chat.so exists!"
fi

# run demo
echo $PWD
python3 python_demo/pipeline.py --model ../../bmodels/qwen-7b_int4_1dev_seq512.bmodel --tokenizer ./support/token_config --devid 0
