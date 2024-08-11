#!/bin/bash
# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/llama3.1-8b_int4_1dev_seq512.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama3.1-8b_int4_1dev_seq512.bmodel
  mv llama3.1-8b_int4_1dev_seq512.bmodel ../../bmodels
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
export PYTHONPATH=$PWD/python_demo:$PYTHONPATH
python3 python_demo/pipeline.py --model ../../bmodels/llama3.1-8b_int4_1dev_seq512.bmodel --tokenizer ./token_config --devid 0
