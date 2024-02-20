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
  docker run --privileged --name mlir -v /dev:/dev -v $PWD:/workspace -it sophgo/tpuc_dev:latest
fi

# download bmodel
if [ $download == "true" ]; then
  pip install dfss
  python3 -m dfss --url=open@sophgo.com:/LLM/LLM-TPU/chatglm3-6b_int4_1dev.bmodel
fi

# compile demo
if [ $compile == "true" ]; then
  cd demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j
  cp chatglm .. && cd ..
fi

# run demo
source /etc/profile.d/libsophon-bin-path.sh
./chatglm --model ../chatglm3-6b_int4_1dev.bmodel --tokenizer ../support/tokenizer.model --devid 0
