#!/bin/bash
set -ex

# download bmodel
if [ ! -d "../../../bmodels" ]; then
  mkdir ../../../bmodels
fi

if [ ! -f "../../../bmodels/llama2-13b_int4_6dev.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama2-13B/llama2-13b_int4_6dev.bmodel
  mv llama2-13b_int4_6dev.bmodel ../../../bmodels
else
  echo "Bmodel Exists!"
fi

if [ ! -f "./llama2_parallel" ]; then
  rm -rf build && mkdir build && cd build
  cmake .. && make -j4
  cp llama2_parallel .. && cd ../
else
  echo "llama2_parallel files Exist!"
fi

./llama2_parallel --model ../../../bmodels/llama2-13b_int4_6dev.bmodel --tokenizer ../token_config/tokenizer.model --devid 16,17,18,19,20,21
