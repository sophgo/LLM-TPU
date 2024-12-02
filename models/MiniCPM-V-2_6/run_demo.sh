#!/bin/bash
# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/minicpmv26_bm1684x_int4_seq1024.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpmv26_bm1684x_int4_seq1024.bmodel
  mv minicpmv26_bm1684x_int4_seq1024.bmodel ../../bmodels
else
  echo "Bmodel Exists!"
fi

if ls ./python_demo/*cpython*.so 1> /dev/null 2>&1; then
  echo "cpython.so exists!"
else
  pushd python_demo
  rm -rf build && mkdir build && cd build
  cmake .. && make
  cp *cpython* ..
  popd
fi

# run demo
echo $PWD
python3 python_demo/pipeline.py --model_path ../../bmodels/minicpmv26_bm1684x_int4_seq1024.bmodel --processor_path ./support/token_config --devid 0
