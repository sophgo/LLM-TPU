#!/bin/bash
set -ex

# 配置环境
pip3 install torch==2.0.1 transformers_stream_generator einops tiktoken accelerate transformers==4.41.2 peft
cp files/Qwen2-7B-Instruct/* /usr/local/lib/python3.10/dist-packages/transformers/models/qwen2/

max_pos_len=10240 # 旋转位置编码的长度，设置为同一个值，才能将block_cache和block权重合并
generation_mode=default # 解码模式
embedding_mode=binary # 设置为binary时，bmodel中不包含embedding，而是放到硬盘
dynamic=1 # prefill阶段开启动态
max_rank_num=64 # 开启lora后，外挂的lora分支的秩
max_embedding_rank_num=64 # 开启lora embedding后，外挂的lora embedding分支的秩

# seq_length_list="10240,8192,7168,6144,5120,4096,3072,2048,1024" # 输入长度 + 输出长度不能超过seq_length
# prefill_length_list="8320,8192,7168,6144,5120,4096,3072,2048,1024" # 输入长度prefill_length
seq_length_list="1024" # 输入长度 + 输出长度不能超过seq_length
prefill_length_list="1024" # 输入长度prefill_length
model_path="/workspace/models/Qwen2-7B-Instruct/" # 训练的pytorch基座模型的路径
lib_path="../share_cache_demo/build/libcipher.so" # 加解密so的路径
lora_config_path="./adapter_config.json" # 微调的lora config的路径
device="cpu"
num_thread=16
tpu_mlir_path="/workspace/tpu-mlir_v1.11.beta.0-65-g1ce2f8ddf-20241029"
tpu_in_pcie="" # --tpu_in_pcie

# Convert comma-separated lists to arrays
IFS=',' read -r -a seq_lengths <<< "$seq_length_list"
IFS=',' read -r -a prefill_lengths <<< "$prefill_length_list"


# 测试单个A16MatMul算子
pushd $tpu_mlir_path
source envsetup.sh
popd
python test_a16matmul.py

# 服务器上有bm1684x的板卡才能跑
for i in "${!seq_lengths[@]}"; do
  seq_length=${seq_lengths[$i]}
  prefill_length=${prefill_lengths[$i]}

  # 测试0~27个block每个block在随机输入情况下，bmodel结果和反量化回torch结果的一致性
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

  # 测试未使用lora+lora_embedding情况下
  # 反量化回torch的完整流程，与bmodel完整流程的一致性
  python test_llm.py \
    --model_path $model_path \
    --device $device \
    --prefill_length $prefill_length \
    --seq_length $seq_length \
    --num_thread $num_thread \
    --max_pos_len $max_pos_len

  # 测试使用lora+lora_embedding情况下
  # 反量化回torch的完整流程，与bmodel完整流程的一致性
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

# 请在soc上测试以下命令
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