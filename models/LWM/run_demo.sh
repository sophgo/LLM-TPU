#!/bin/bash
# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/lwm-text-chat-1m_int4_1dev_seq512.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/lwm-text-chat-1m_int4_1dev_seq512.bmodel
  mv lwm-text-chat-1m_int4_1dev_seq512.bmodel ../../bmodels
else
  echo "Bmodel Exists!"
fi

if [ ! -f "./demo/lwm" ]; then
  cd demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j4
  cp lwm ..
  cd ../..
else
  echo "lwm file Exists!"
fi

# run demo
echo $PWD
./demo/lwm --model ../../bmodels/lwm-text-chat-1m_int4_1dev_seq512.bmodel --tokenizer ./support/tokenizer.model --devid 0
