#!/bin/bash
set -ex

# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/qwen-7b_int4_1dev_none_addr.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/LLM/LLM-TPU/qwen-7b_int4_1dev_none_addr.bmodel
  mv qwen-7b_int4_1dev_none_addr.bmodel ../../bmodels
else
  echo "Bmodel Exists!"
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
  cmake .. && make -j4
  cp qwen .. && cd ../..
else
  git submodule update --init
  echo "qwen files Exist!"
fi

# run demo
# source /etc/profile.d/libsophon-bin-path.sh
# export LD_LIBRARY_PATH=$PWD/../libsophon-0.5.0/lib
./demo/qwen --model ../../bmodels/qwen-7b_int4_1dev_none_addr.bmodel --tokenizer ./support/qwen.tiktoken --devid 0
