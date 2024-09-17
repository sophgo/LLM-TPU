#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: \$0 <mode> <seq_length>"
    exit 1
fi

mode=$1
seq_length=$2

set -e

echo "Install the required Python lib..."
pip3 install -r requirements.txt
pip3 install modelscope

echo "Download model..."
python3 -c "from modelscope import snapshot_download; snapshot_download('LLM-Research/Meta-Llama-3.1-8B-Instruct', local_dir='./origin_model')"

echo "Replace the files in the transformers lib..."
pkg_path=$(pip show transformers | grep Location | cut -d ' ' -f2)
cp ${pkg_path}/transformers/models/llama/modeling_llama.py modeling_llama_backup.py
sudo cp ./compile/files/Meta-Llama-3.1-8B-Instruct/modeling_llama.py ${pkg_path}/transformers/models/llama/modeling_llama.py

echo "export onnx..."
cd compile
python export_onnx.py --model_path ../origin_model --seq_length ${seq_length}

echo "compile model..."
pushd /workspace/tpu-mlir && source envsetup.sh && ./build.sh DEBUG && popd
source ./compile.sh --mode ${mode} --name llama3.1-8b --seq_length ${seq_length}
echo "compile model success"

echo "all done"
