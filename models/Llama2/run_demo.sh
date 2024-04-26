#!/bin/bash
# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/llama2-7b_int4_1dev.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama2-7b_int4_1dev.bmodel
  mv llama2-7b_int4_1dev.bmodel ../../bmodels
else
  echo "Bmodel Exists!"
fi

if [ ! -f "./python_demo/chat.cpython-310-x86_64-linux-gnu.so" ]; then
  cd python_demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j
  cp chat.cpython-310-x86_64-linux-gnu.so ..
  cd ../..
else
  echo "chat.so exists!"
fi

# run demo
echo $PWD
python3 python_demo/pipeline.py --model ../../bmodels/llama2-7b_int4_1dev.bmodel --tokenizer ./support/token_config --devid 0
