//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "bmruntime_interface.h"
#include "memory.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <inttypes.h>
#include <iostream>
#include <numeric>
#include <random>
#include <stdio.h>
#include <string>
#include <vector>

typedef std::vector<int> ArrayInt;
typedef std::vector<float> ArrayFloat;

class InternVL3 {
public:
  void init(const std::vector<int> &devid, std::string model_path);
  void deinit();
  void forward_embed(ArrayInt const &tokens);
  void forward_vit(ArrayFloat const &pixel_values, int vit_offset);
  int forward_first();
  int forward_next();
  void clear_history();

  std::mt19937 sgen;
  InternVL3() : sgen(std::random_device()()) {};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  void net_launch_block_dyn(const bm_net_info_t *net, int real_len);
  void net_launch_kv_dyn(const bm_net_info_t *net, int real_len, int kv_len);
  void net_launch_decode(int block_idx, int kv_offset,
                         bm_device_mem_t &input_mem, const int *position_id,
                         std::vector<uint16_t> &attention_mask);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  void init_by_names();
  int forward_first_with_kv();
  int greedy_search(bm_device_mem_t &logits_mem);
  int penalty_sample(bm_device_mem_t &logits_mem);

public:
  int token_length;
  int history_length;
  int SEQLEN;
  int HIDDEN_SIZE;
  int KV_BYTES;
  int NUM_LAYERS;
  int NUM_IMAGE_TOKEN;
  int MAX_INPUT_LENGTH;
  int PREFILL_KV_LENGTH;
  bool prefill_mask;
  uint16_t mask_value;
  bool lmhead_with_topk;
  bool support_history;
  std::vector<int> visited_tokens;
  bool is_dynamic;

  // generation
  std::string generation_mode;
  float penalty;
  float temperature;
  int top_k;
  float top_p;

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  std::vector<const bm_net_info_t *> net_blocks_kv;
  const bm_net_info_t *net_vit;
  const bm_net_info_t *net_embed, *net_embed_cache;
  const bm_net_info_t *net_lm, *net_greedy_head, *net_sample_head;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};
