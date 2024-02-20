#!/bin/bash
set -ex
docker=
download=
compile=

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
  docker run --privileged --name mlir -v /dev:/dev -v $PWD:/workspace -it sophgo/tpuc_dev:latest bash
  docker exec -it mlir bash
fi

# download bmodel
if [ $download == "true" ]; then
  pip install dfss
  python3 -m dfss --url=open@sophgo.com:/LLM/LLM-TPU/llama2-7b_int4_1dev.bmodel
fi

# compile demo
if [ $compile == "true" ]; then
  cd demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j
  cp llama2 .. && cd ..
fi

# run demo
source /etc/profile.d/libsophon-bin-path.sh
./llama2 --model ../llama2-7b_int4_1dev.bmodel --tokenizer ../support/tokenizer.model --devid 0
