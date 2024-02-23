#!/bin/bash
set -ex
docker=
download=
compile=
arch=

if [ ! -d "models" ]; then
  mkdir models
fi

if [ ! -f "./models/qwen-7b_int4_1dev.bmodel" ]; then
  pip install dfss
  python3 -m dfss --url=open@sophgo.com:/LLM/LLM-TPU/qwen-7b_int4_1dev.bmodel
  mv qwen-7b_int4_1dev.bmodel ./models
else
  echo "Model Exists!"
fi

# download libsophon
# if [ $arch == "pcie" ]; then
#   python3 -m dfss --url=open@sophgo.com:/LLM/libsophon-0.5.0_pcie.tar.gz
#   tar xvf libsophon-0.5.0_pcie.tar.gz
# elif [ $arch = "soc" ]; then 
#   python3 -m dfss --url=open@sophgo.com:/LLM/libsophon-0.5.0_soc.tar.gz
#   tar xvf libsophon-0.5.0_soc.tar.gz
# fi

if [ ! -f "./demo/qwen" ]; then
  git submodule update --init
  cd demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j
  cp qwen .. && cd ..
else
  git submodule update --init
  cd ./demo
  echo "qwen file Exists!"
fi

# run demo
# source /etc/profile.d/libsophon-bin-path.sh
# export LD_LIBRARY_PATH=$PWD/../libsophon-0.5.0/lib
./qwen --model ../models/qwen-7b_int4_1dev.bmodel --tokenizer ../support/qwen.tiktoken --devid 0
