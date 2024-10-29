#!/bin/bash
set -ex

python3 pipeline.py \
  --model_path encrypted.bmodel  \
  --tokenizer_path ../support/token_config/ \
  --devid 10 --generation_mode penalty_sample \
  --lib_path ../share_cache_demo/build/libcipher.so \
  --embedding_path embedding.bin \
  --lora_path encrypted_lora_weights.bin \
  --enable_lora_embedding