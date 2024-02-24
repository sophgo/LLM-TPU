#!/bin/bash
set -ex
model=
demo=
arch=

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --model)
            model="$2"
            shift 2
            ;;
        --arch)
            arch="$2"
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

# model
if [ $model == "chatglm3-6b" ]; then
  demo="chatglm"
elif [ $model = "llama2-7b" ]; then 
  demo="llama2"
elif [ $model = "qwen-7b" ]; then 
  demo="qwen"
else
  >&2 echo -e "Error: Invalid name $model, the input name must be \033[31mchatglm3-6b|llama2-7b|qwen-7b\033[0m"
  exit 1
fi

# arch 
if [ $arch == "pcie" ]; then
  demo=$demo"_pcie"
elif [ $arch = "soc" ]; then 
  demo=demo"_soc"
else
  >&2 echo -e "Error: Invalid name $arch, the input name must be \033[31mpcie|soc\033[0m"
  exit 1
fi

# mkdir
if [ ! -d "deploy" ]; then
  mkdir deploy
fi

# begin download
pushd deploy

# download bmodel
if [ ! -f $model'_int4_1dev.bmodel' ]; then
  pip install dfss
  python3 -m dfss --url=open@sophgo.com:/LLM/LLM-TPU/$model\_int4_1dev.bmodel
else
  echo "Model Exists!"
fi

# download libsophon
if [ $arch == "pcie" ] && [ ! -f 'libsophon-0.5.0_pcie.tar.gz' ]; then
  python3 -m dfss --url=open@sophgo.com:/LLM/LLM-TPU/libsophon-0.5.0_pcie.tar.gz
  tar xvf libsophon-0.5.0_pcie.tar.gz
elif [ $arch = "soc" ] && [ ! -f 'libsophon-0.5.0_soc.tar.gz' ]; then 
  python3 -m dfss --url=open@sophgo.com:/LLM/LLM-TPU/libsophon-0.5.0_soc.tar.gz
  tar xvf libsophon-0.5.0_soc.tar.gz
fi

# download exe file
if [ ! -f "$demo" ]; then
  python3 -m dfss --url=open@sophgo.com:/LLM/LLM-TPU/$demo
else
  echo "$demo file Exists!"
fi

# env
chmod -R +777 .
export LD_LIBRARY_PATH=$PWD/libsophon-0.5.0/lib

# run demo
if [ $model == "chatglm3-6b" ]; then
  ./$demo --model ./$model\_int4_1dev.bmodel --tokenizer ../models/ChatGLM3/support/tokenizer.model --devid 0
elif [ $model = "llama2-7b" ]; then 
  ./$demo --model ./$model\_int4_1dev.bmodel --tokenizer ../models/Llama2/support/tokenizer.model --devid 0
elif [ $model = "qwen-7b" ]; then 
  ./$demo --model ./$model\_int4_1dev.bmodel --tokenizer ../models/Qwen/support/qwen.tiktoken --devid 0
fi

popd