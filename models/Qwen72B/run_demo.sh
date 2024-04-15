#!/bin/bash
set -ex

# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/qwen-72b_int4_8192_8dev.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/qwen-72b/qwen-72b_int4_8192_8dev.bmodel
  mv qwen-72b_int4_8192_8dev.bmodel ../../bmodels
else
  echo "Bmodel Exists!"
fi

if [ ! -f "./demo/qwen" ]; then
  git submodule update --init
  cd demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j4
  cp qwen .. && cd ../..
else
  git submodule update --init
  echo "qwen files Exist!"
fi

# ./demo/qwen --model ../../bmodels/qwen-72b_int4_8192_8dev.bmodel --tokenizer ./support/qwen.tiktoken --devid 0,1,2,3,4,5,6,7
./demo/qwen --model ../../bmodels/qwen-72b_int4_8192_8dev.bmodel --tokenizer ./support/qwen.tiktoken --devid 16,17,18,19,20,21,22,23
