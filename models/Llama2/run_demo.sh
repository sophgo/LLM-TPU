#!/bin/bash
# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/llama2-7b_int4_1dev.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/LLM/LLM-TPU/llama2-7b_int4_1dev.bmodel
  mv llama2-7b_int4_1dev.bmodel ../../bmodels
else
  echo "Bmodel Exists!"
fi

if [ ! -f "./demo/llama2" ]; then
  cd demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j
  cp llama2 ..
  cd ../..
else
  echo "llama2 file Exists!"
fi

# run demo
echo $PWD
./demo/llama2 --model ../../bmodels/llama2-7b_int4_1dev.bmodel --tokenizer ./support/tokenizer.model --devid 0
