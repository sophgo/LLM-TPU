#!/bin/bash
# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/llama3-8b_int4_1dev_512_addr_mode.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama3-8b_int4_1dev_512_addr_mode.bmodel
  mv llama3-8b_int4_1dev_512_addr_mode.bmodel ../../bmodels
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
export PYTHONPATH=$PWD/python_demo:$PYTHONPATH
python3 python_demo/pipeline.py --model ../../bmodels/llama3-8b_int4_1dev_512_addr_mode.bmodel --tokenizer ./token_config --devid 0
