#!/bin/bash
set -ex
docker=
download=
compile=
arch=

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --docker)
            docker="true"
            shift
            ;;
        --download)
            download="true"
            shift
            ;;
        --compile)
            compile="true"
            shift
            ;;
        --arch)
            compile="$2"
            shift 2
            ;;
        *)
            echo "Invalid option: $key" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

# install docker
if [ $docker == "true" ]; then
  docker pull sophgo/tpuc_dev:latest
  docker run --privileged --name mlir -v /dev:/dev -v $PWD:/workspace -it sophgo/tpuc_dev:latest
fi

# download bmodel
if [ $download == "true" ]; then
  pip install dfss
  python3 -m dfss --url=open@sophgo.com:/LLM/LLM-TPU/qwen-7b_int4_1dev.bmodel
fi

# download libsophon
if [ $arch == "pcie" ]; then
  python3 -m dfss --url=open@sophgo.com:/LLM/libsophon-0.5.0_pcie.tar.gz
  tar xvf libsophon-0.5.0_pcie.tar.gz
elif [ $arch = "soc" ]; then 
  python3 -m dfss --url=open@sophgo.com:/LLM/libsophon-0.5.0_soc.tar.gz
  tar xvf libsophon-0.5.0_soc.tar.gz
fi

# compile demo
if [ $compile == "true" ]; then
  git submodule update --init
  cd demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j
  cp qwen .. && cd ..
fi

# run demo
source /etc/profile.d/libsophon-bin-path.sh
export LD_LIBRARY_PATH=$PWD/../libsophon-0.5.0/lib
./qwen --model ../qwen-7b_int4_1dev.bmodel --tokenizer ../support/qwen.tiktoken --devid 0
