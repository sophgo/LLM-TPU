#!/bin/bash
set -ex

# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/qwen2.5-7b_int4_seq8192_1dev.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-7b_int4_seq8192_1dev.bmodel
  mv qwen2.5-7b_int4_seq8192_1dev.bmodel ../../bmodels
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

# run demo
echo $PWD
python3 python_demo/pipeline.py --model ../../bmodels/qwen2.5-7b_int4_seq8192_1dev.bmodel --tokenizer ./support/token_config --devid 0
