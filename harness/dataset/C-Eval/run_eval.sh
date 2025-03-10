#!/bin/bash
# set -ex

devid="8"
model="chatglm3"
bmodel_dir="../../bmodels"
tokenizer_path=""

# 解析参数
for (( i=1; i<=$#; i++ )); do
    eval opt='$'{${i}}
    case ${opt} in
        --devid)
            let "i++"
            eval devid='$'{${i}}
            ;;
        --model)
            let "i++"
            eval model='$'{${i}}
            ;;
        --bmodel_dir)
            let "i++"
            eval bmodel_dir='$'{${i}}
            ;;
        --tokenizer_path)
            let "i++"
            eval tokenizer_path='$'{${i}}
            ;;
        
        *)
            echo "Error : invalid opt ${opt}"
            exit 1
    esac
done

# download c-eval dataset
if [ ! -d "./ceval-exam" ]; then
  mkdir ceval-exam 
  cd ceval-exam
  wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
  unzip ceval-exam
  cd ../
else
  echo "Dataset Exists!"
fi

case $model in
    "chatglm3")
        bmodel_name="chatglm3-6b_int4_1dev.bmodel"
        model="chatglm3"
        eval_mode="fast"
        if [ ! -e "/path/to/file" ]; then
            tokenizer_path="../../models/ChatGLM3/support/token_config/"
        fi
        ;;
    "qwen1.5")
        bmodel_name="qwen1.5-1.8b_f16_seq1280_1dev.bmodel"
        model="qwen1_5"
        eval_mode="default"
        if [ ! -e "/path/to/file" ]; then
            tokenizer_path="../../models/Qwen1_5/token_config/"
        fi
        ;;
    *)
        echo "Unknown model name!"
        exit 1
        ;;
esac

# download bmodel
if [ ! -d $bmodel_dir ]; then
  mkdir $bmodel_dir
  echo "Create model directory.!"
fi

whole_model_path="$bmodel_dir/$bmodel_name"

if [ ! -f $whole_model_path ]; then
  download_command="python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/$bmodel_name"
  pip3 install dfss
  $download_command
  mv $bmodel_name $bmodel_dir
else
  echo "Bmodel Exists!"
fi

# run demo
echo $PWD
export PYTHONPATH=../../
python evaluate_$model.py --devid $devid --model_path $whole_model_path --tokenizer_path $tokenizer_path --eval_mode $eval_mode