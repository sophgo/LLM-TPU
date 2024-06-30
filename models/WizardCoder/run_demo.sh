#!/bin/bash
# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/wizardcoder-15b_int4_1dev_seq512.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/wizardcoder-15b_int4_1dev_seq512.bmodel
  mv wizardcoder-15b_int4_1dev_seq512.bmodel ../../bmodels
else
  echo "Bmodel Exists!"
fi

if [ ! -f "./demo/wizardcoder" ]; then
  cd demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j4
  cp wizardcoder ..
  cd ../..
else
  echo "wizardcoder file Exists!"
fi

# run demo
echo $PWD
./demo/wizardcoder --model ../../bmodels/wizardcoder-15b_int4_1dev_seq512.bmodel --vocab ./vocab/vocab.json --devid 0
