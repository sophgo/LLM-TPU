#!/bin/bash
set -ex

# 默认参数设置
model_path=""
tpu_mlir_path=""
mode="int4"
seq_length=
model_name=""
model_name_upper=""
num_device=1
config_file="../../../config.json"
tpu_mlir_name=""

read_config() {
    python3 -c "import json; print(json.load(open('${config_file}'))['tpu_mlir_name'])"
}

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
    --num_device)
        num_device="$2"
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

if [ "$model_name" = "llama3.2-11b" ]; then
  model_name_upper="Llama-3.2-11B-Vision"
  echo "Compile Llama-3.2-11B-Vision"
else
  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mllama3.2-11b\033[0m"
  exit 1
fi

echo "Install the required Python lib..."
pip install transformers_stream_generator einops tiktoken accelerate transformers==4.45.2
pip install torch==2.2.0+cpu torchvision --index-url https://download.pytorch.org/whl/cpu

# 根据 model_path 的值决定是否下载模型
if [[ -z "$model_path" ]]; then
    pip3 install modelscope
    echo "Download model..."
    python3 -c "from modelscope import snapshot_download; snapshot_download('LLM-Research/Llama-3.2-11B-Vision-Instruct', local_dir='./${model_name_upper}-Instruct')"
    model_path="./${model_name_upper}-Instruct"
fi

if [[ -z "$tpu_mlir_path" ]]; then
    tpu_mlir_name=$(read_config "$config_file" "tpu_mlir_name")
    pip3 install dfss
    python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/mlir_club/${tpu_mlir_name}.tar.gz
    tar -xf ${tpu_mlir_name}.tar.gz
    tpu_mlir_path="./${tpu_mlir_name}"
fi

echo "Replace the files in the transformers lib..."
pkg_path=$(pip show transformers | grep Location | cut -d ' ' -f2)
cp ${pkg_path}/transformers/models/mllama/modeling_mllama.py modeling_mllama_backup.py
sudo cp files/${model_name_upper}-Instruct/modeling_mllama.py ${pkg_path}/transformers/models/mllama/modeling_mllama.py

echo "export onnx..."
if [ x$num_device != x1 ]; then
    python export_onnx.py --model_path ${model_path} --seq_length ${seq_length} --lmhead_with_topk 1
else
    python export_onnx.py --model_path ${model_path} --seq_length ${seq_length}
fi

echo "compile model..."
source ${tpu_mlir_path}/envsetup.sh 
source ./compile.sh --mode ${mode} --name ${model_name} --seq_length ${seq_length} --addr_mode io_alone --num_device ${num_device}
echo "compile model success"

echo "all done"
