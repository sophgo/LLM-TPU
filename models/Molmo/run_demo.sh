#!/bin/bash
# download bmodel
if [ ! -d "../../bmodels" ]; then
  mkdir ../../bmodels
fi

if [ ! -f "../../bmodels/molmo-7b_int4_seq1024_384x384.bmodel" ]; then
  pip3 install dfss
  python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/molmo-7b_int4_seq1024_384x384.bmodel
  mv molmo-7b_int4_seq1024_384x384.bmodel ../../bmodels
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
export PYTHONPATH=$PWD/python_demo:$PYTHONPATH
python3 python_demo/pipeline.py --model ../../bmodels/molmo-7b_int4_seq1024_384x384.bmodel --image_path python_demo/test.jpg  --image_size 384 --processor_path ./processor_config --devid 0
