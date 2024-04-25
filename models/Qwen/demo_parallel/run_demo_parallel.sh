#!/bin/bash
set -ex

model=qwen-72b_int4_8192_8dev.bmodel

# download bmodel
if [ ! -d "../../../bmodels" ]; then
  mkdir ../../../bmodels
fi

if [ ! -f "../../../bmodels/${model}" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/bmodels/${model}
  mv ${model} ../../../bmodels
else
  echo "Bmodel Exists!"
fi

if [ ! -f "./qwen_parallel" ]; then
  git submodule update --init
  rm -rf build && mkdir build && cd build
  cmake .. && make -j4
  cp qwen_parallel .. && cd ../
else
  git submodule update --init
  echo "qwen_parallel files Exist!"
fi

./qwen_parallel --model ../../../bmodels/${model} --tokenizer ../token_config/qwen.tiktoken --devid 0,1,2,3,4,5,6,7
