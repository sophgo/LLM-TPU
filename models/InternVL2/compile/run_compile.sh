#!/bin/bash
set -ex
mode=int4
name="internvl2-4b"
tpu_mlir_path="/workspace/tpu-mlir/"

if [ -d "${tpu_mlir_path}" ]; then
    pushd ${tpu_mlir_path} && source envsetup.sh && ./build.sh DEBUG && popd
else
    echo "cannot access ${tpu_mlir_path}: No such directory!"
    echo "Please provide a valid tpu-mlir path firstly!"
    exit 1
fi

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    --name)
        name="$2"
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

set -e

echo "Install the required Python lib..."
pip install transformers==4.37.2
pip3 install transformers_stream_generator einops tiktoken accelerate timm sentencepiece

sudo cp ./files/${name}/*.py ${model_path}

echo "export onnx..."
python export_onnx.py --model_path ${model_path}

echo "compile model..."

case ${name} in
    "InternVL2-2B")
        compile_name="internvl2-2b"
        ;;
    "InternVL2-4B")
        compile_name="internvl2-4b"
        ;;
    *)
        echo "Invalid name ${name}, the input name must be \033[31mInternVL2-4B|InternVL2-2B\033[0m"
        exit 1
        ;;
esac

./compile.sh --mode ${mode} --name ${compile_name}
echo "compile model success"

echo "all done"