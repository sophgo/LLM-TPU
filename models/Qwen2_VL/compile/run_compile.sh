#!/bin/bash

set -e

# 默认参数设置
model_path=""
tpu_mlir_path=""
mode="int4"
seq_length=
model_name=""
model_name_upper=""


# 参数解析
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    --model_name)
        model_name="$2"
        shift 2
        ;;
    --model_path)
        model_path="$2"
        shift 2
        ;;
    --mode)
        mode="$2"
        shift 2
        ;;
    --seq_length)
        seq_length="$2"
        shift 2
        ;;
    --tpu_mlir_path)
        tpu_mlir_path="$2"
        shift 2
        ;;
    *)
        echo "Invalid option: $key" >&2
        exit 1
        ;;
    esac
done

# 检查必要参数
if [[ -z "$model_name" ]]; then
    echo "Model name is required."
    exit 1
fi
if [[ -z "$seq_length" ]]; then
    echo "Error: --seq_length is required." >&2
    exit 1
fi

if [ "$model_name" = "qwen2-vl-7b" ]; then
  model_name_upper="Qwen2-VL-7B"
  echo "Compile Qwen2-VL-7B"
elif [ "$model_name" = "qwen2-vl-2b" ]; then
  model_name_upper="Qwen2-VL-2B"
  echo "Compile Qwen2-VL-2B"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mqwen2-vl-7b|qwen2-vl-2b\033[0m"
  exit 1
fi

echo "Install the required Python lib..."
pip install qwen-vl-utils accelerate torch==2.5.0 transformers==4.45.1
pip3 install dfss
pip3 install modelscope

# 根据 model_path 的值决定是否下载模型
if [[ -z "$model_path" ]]; then
    echo "Download model..."
    python3 -c "from modelscope import snapshot_download; snapshot_download('Qwen/${model_name_upper}-Instruct', local_dir='./${model_name_upper}-Instruct')"
    model_path="../${model_name_upper}-Instruct"
fi

if [[ -z "$tpu_mlir_path" ]]; then
    python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/mlir_club/tpu-mlir_v1.9.beta.0-84-ga12293f84-20240921.tar.gz
    tar -xf tpu-mlir_v1.9.beta.0-84-ga12293f84-20240921.tar.gz
    tpu_mlir_path="../tpu-mlir_v1.9.beta.0-84-ga12293f84-20240921"
fi

echo "Replace the files in the transformers lib..."
pkg_path=$(pip show transformers | grep Location | cut -d ' ' -f2)
cp ${pkg_path}/transformers/models/qwen2_vl/modeling_qwen2_vl.py modeling_qwen2_vl_backup.py
sudo cp files/${model_name_upper}-Instruct/modeling_qwen2_vl.py ${pkg_path}/transformers/models/qwen2/modeling_qwen2_vl.py

echo "export onnx..."
python export_onnx.py --model_path ${model_path} --seq_length ${seq_length}

echo "compile model..."
source ${tpu_mlir_path}/envsetup.sh 
source ./compile.sh --mode ${mode} --name ${model_name} --seq_length ${seq_length} --addr_mode io_alone
echo "compile model success"

echo "all done"