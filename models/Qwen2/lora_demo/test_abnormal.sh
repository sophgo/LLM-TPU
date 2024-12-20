#!/bin/bash
set -ex
mode=

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
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

# compile
if [ "$mode" == "compile" ]; then
  pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
  pip3 install transformers_stream_generator einops tiktoken accelerate transformers==4.41.2 peft

  max_rank_num=64 # 开启lora后，外挂的lora分支的秩
  max_embedding_rank_num=64 # 开启lora embedding后，外挂的lora embedding分支的秩

  model_path="/workspace/models/Qwen2-7B-Instruct/" # 训练的pytorch基座模型的路径
  lib_path="../share_cache_demo/build/libcipher.so" # 加解密so的路径
  lora_path="saves_lora/lora_sft_qwen2_unpretrained_init/" # 微调的lora模型的路径
  lora_embedding_path="saves_lora/lora_sft_qwen2_unpretrained_init_embedding/" # 微调的lora模型的路径
  device="cpu"
  num_thread=16

  python export_abnormal.py \
    --model_path $model_path \
    --device $device \
    --num_thread $num_thread \
    --lib_path $lib_path \
    --lora_path $lora_path \
    --lora_embedding_path $lora_embedding_path \
    --max_rank_num $max_rank_num \
    --max_embedding_rank_num $max_embedding_rank_num

elif [ "$mode" == "run" ]; then
  pushd test_abnormal
  touch embedding.bin.empty
  touch encrypted_lora_weights.bin.empty


  python3 test_abnormal.py \
    --model_path /data2/v4/encrypted.bmodel \
    --tokenizer_path ../support/token_config/ \
    --devid 0 \
    --generation_mode greedy \
    --lib_path ../share_cache_demo/build/libcipher.so \
    --abnormal_path ./test_abnormal \
    --enable_lora_embedding | tee test_abnormal.log
else
  echo "Error: Unknown mode '$mode'. Use 'compile' or 'run'."
  exit 1
fi