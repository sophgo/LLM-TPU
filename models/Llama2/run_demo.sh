#!/bin/bash
# download bmodel
if [ ! -d "models" ]; then
  mkdir models
fi

if [ ! -f "./models/llama2-7b_int4_1dev.bmodel" ]; then
  echo $PWD
  pip install dfss
  python3 -m dfss --url=open@sophgo.com:/LLM/LLM-TPU/llama2-7b_int4_1dev.bmodel
  mv llama2-7b_int4_1dev.bmodel ./models
else
  echo "Model Exists!"
fi

if [ ! -f "./demo/llama2" ]; then
  cd demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j
  cp llama2 .. && cd ..
else
  echo "llama2 file Exists!"
fi

# run demo
source /etc/profile.d/libsophon-bin-path.sh
./demo/llama2 --model ./models/llama2-7b_int4_1dev.bmodel --tokenizer ./support/tokenizer.model --devid 0
