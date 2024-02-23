#!/bin/bash
set -ex
docker=
download=
compile=

#!/bin/bash
# download bmodel
if [ ! -d "models" ]; then
  mkdir models
fi

if [ ! -f "./models/chatglm3-6b_int4_1dev.bmodel" ]; then
  echo $PWD
  pip install dfss
  python3 -m dfss --url=open@sophgo.com:/LLM/LLM-TPU/chatglm3-6b_int4_1dev.bmodel
  mv chatglm3-6b_int4_1dev.bmodel ./models
else
  echo "Model Exists!"
fi

if [ ! -f "./demo/chatglm" ]; then
  cd demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j
  cp chatglm .. && cd ..
else
  echo "chatglm file Exists!"
fi

# run demo
source /etc/profile.d/libsophon-bin-path.sh
./demo/chatglm --model ./models/chatglm3-6b_int4_1dev.bmodel --tokenizer ./support/tokenizer.model --devid 0
