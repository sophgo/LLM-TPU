#!/bin/bash
# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/llama3-8b_int4_1dev_seq512.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama3-8b_int4_1dev_seq512.bmodel
  mv llama3-8b_int4_1dev_seq512.bmodel ../../bmodels
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
export PYTHONPATH=$PWD/python_demo:$PYTHONPATH
python3 python_demo/pipeline.py --model ../../bmodels/llama3-8b_int4_1dev_seq512.bmodel --tokenizer ./token_config --devid 0
