#!/bin/bash
set -ex

# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/qwen1.5-1.8b_int4_1dev.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen1.5-1.8b_int4_1dev.bmodel
  mv qwen1.5-1.8b_int4_1dev.bmodel ../../bmodels
else
  echo "Bmodel Exists!"
fi

# download libsophon
# if [ $arch == "pcie" ]; then
#   python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/libsophon-0.5.0_pcie.tar.gz
#   tar xvf libsophon-0.5.0_pcie.tar.gz
# elif [ $arch = "soc" ]; then 
#   python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/libsophon-0.5.0_soc.tar.gz
#   tar xvf libsophon-0.5.0_soc.tar.gz
# fi

if [ ! -f "./python_demo/chat.cpython-310-x86_64-linux-gnu.so" ]; then
  cd python_demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j4
  cp chat.cpython-310-x86_64-linux-gnu.so .. && cd ../..
else
  echo "qwen1.5 files Exist!"
fi

# run demo
# source /etc/profile.d/libsophon-bin-path.sh
# export LD_LIBRARY_PATH=$PWD/../libsophon-0.5.0/lib
source ../../envsetup.sh
python3 python_demo/chat.py --model_path ../../bmodels/qwen1.5-1.8b_int4_1dev.bmodel --tokenizer_path ./support/token_config --devid '0'
