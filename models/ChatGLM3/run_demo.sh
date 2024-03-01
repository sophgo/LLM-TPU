#!/bin/bash
set -ex

#!/bin/bash
# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/chatglm3-6b_int4_1dev.bmodel" ]; then
  pip install dfss
  python3 -m dfss --url=open@sophgo.com:/LLM/LLM-TPU/chatglm3-6b_int4_1dev.bmodel
  mv chatglm3-6b_int4_1dev.bmodel ../../bmodels
else
  echo "Bmodel Exists!"
fi

if [ ! -f "./demo/chatglm" ]; then
  cd demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j
  cp chatglm .. && cd ../..
else
  echo "chatglm file Exists!"
fi

# run demo
./demo/chatglm --model ../../bmodels/chatglm3-6b_int4_1dev.bmodel --tokenizer ./support/tokenizer.model --devid 0
