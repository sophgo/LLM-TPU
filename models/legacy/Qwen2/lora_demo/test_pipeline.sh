#!/bin/bash
set -ex

# Set up the environment
pip3 install torch==2.0.1 transformers_stream_generator einops tiktoken accelerate transformers==4.41.2 peft
cp files/Qwen2-7B-Instruct/* /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/

max_pos_len=10240 # length of the rotary position embedding; must be set to the same value so that block_cache and block weights can be merged
generation_mode=default # decoding mode
embedding_mode=binary # when set to binary, the bmodel does not contain the embedding; it is stored on disk instead
dynamic=1 # enable dynamic shapes in the prefill stage
max_rank_num=64 # rank of the external lora branch when lora is enabled
max_embedding_rank_num=64 # rank of the external lora embedding branch when lora embedding is enabled

# seq_length_list="10240,8192,7168,6144,5120,4096,3072,2048,1024" # input length + output length must not exceed seq_length
# prefill_length_list="8320,8192,7168,6144,5120,4096,3072,2048,1024" # input length prefill_length
seq_length_list="1024" # input length + output length must not exceed seq_length
prefill_length_list="1024" # input length prefill_length
model_path="/workspace/models/Qwen2-7B-Instruct/" # path to the trained pytorch base model
lib_path="../share_cache_demo/build/libcipher.so" # path to the encryption/decryption .so
lora_config_path="./adapter_config.json" # path to the fine-tuned lora config
device="cpu"
num_thread=16
tpu_mlir_path="/workspace/tpu-mlir_v1.11.beta.0-65-g1ce2f8ddf-20241029"
tpu_in_pcie="" # --tpu_in_pcie

# Convert comma-separated lists to arrays
IFS=',' read -r -a seq_lengths <<< "$seq_length_list"
IFS=',' read -r -a prefill_lengths <<< "$prefill_length_list"


# Test a single A16MatMul operator
pushd $tpu_mlir_path
source envsetup.sh
popd
python test_a16matmul.py

# requires a server with a bm1684x board to run
for i in "${!seq_lengths[@]}"; do
  seq_length=${seq_lengths[$i]}
  prefill_length=${prefill_lengths[$i]}

  # Test the consistency between the bmodel result and the dequantized-to-torch result for each of blocks 0~27 with random inputs
  if [[ -n "$tpu_in_pcie" ]]; then
    export USING_CMODEL=False
    export LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib/:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$PWD/../support/lib_pcie:$LD_LIBRARY_PATH
  fi
  python test_block.py \
    --model_path $model_path \
    --device $device \
    --prefill_length $prefill_length \
    --seq_length $seq_length \
    --num_thread $num_thread \
    --max_pos_len $max_pos_len \
    $tpu_in_pcie
done


pip3 uninstall transformers -y
pip3 install transformers==4.41.2
rm -rf /root/.cache/tpu-mlir

for i in "${!seq_lengths[@]}"; do
  seq_length=${seq_lengths[$i]}
  prefill_length=${prefill_lengths[$i]}

  # Test the consistency between the full dequantized-to-torch flow and the full bmodel flow
  # without lora+lora_embedding
  python test_llm.py \
    --model_path $model_path \
    --device $device \
    --prefill_length $prefill_length \
    --seq_length $seq_length \
    --num_thread $num_thread \
    --max_pos_len $max_pos_len

  # Test the consistency between the full dequantized-to-torch flow and the full bmodel flow
  # with lora+lora_embedding
  python test_lora.py \
      --model_path $model_path \
      --device $device \
      --prefill_length $prefill_length \
      --seq_length $seq_length \
      --num_thread $num_thread \
      --max_pos_len $max_pos_len \
      --lib_path $lib_path \
      --lora_config_path $lora_config_path \
      --max_rank_num $max_rank_num \
      --max_embedding_rank_num $max_embedding_rank_num
done

# Test the following commands on SoC
# mkdir third_party && cd third_party
# git clone https://github.com/rogersce/cnpy.git
# cd ..
# rm -rf build && mkdir build
# cd build && cmake -DCMAKE_TYPE=DUMP .. && make && cp *cpython* .. && cd ..

# mkdir test_lora
# mv /path_to/*encrypted_lora_weights.bin test_lora
# mv /path_to/*torch_hidden_states.npy test_lora
# python3 test_pipeline.py \
#     --model_path encrypted.bmodel \
#     --tokenizer_path ../support/token_config/ \
#     --devid 0 \
#     --generation_mode greedy \
#     --lib_path ../share_cache_demo/build/libcipher.so \
#     --embedding_path embedding.bin \
#     --lora_path encrypted_lora_weights.bin \
#     --enable_lora_embedding