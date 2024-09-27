#!/bin/bash
# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/llama2-7b_int4_1dev_seq512.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama2-7b_int4_1dev_seq512.bmodel
  mv llama2-7b_int4_1dev_seq512.bmodel ../../bmodels
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
python3 python_demo/pipeline.py --model ../../bmodels/llama2-7b_int4_1dev_seq512.bmodel --tokenizer ./support/token_config --devid 0
