//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "bmruntime_interface.h"
#include "memory.h"
#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <inttypes.h>
#include <iostream>
#include <numeric>
#include <random>
#include <stdio.h>
#include <string>
#include <vector>

class Qwen {
public:
  void init(std::string model_path, std::string config_path, bool do_sample,
            const std::vector<int> &devid, int rep_window);
  void deinit();
  int forward_first(std::vector<int> &tokens);
  int forward_next();
  void clear_kv();
  bool check_stop(const std::string &text);

  std::mt19937 sgen;
  Qwen() : sgen(std::random_device()()){};

private:
  int forward_first_with_kv(std::vector<int> &tokens);
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  void net_launch_dyn(const bm_net_info_t *net, int real_len,
                      int stage_idx = 0);
  void net_launch_kv_dyn(const bm_net_info_t *net, int real_len, int kv_len);
  void net_launch_decode(int block_idx, int kv_offset,
                         bm_device_mem_t &input_mem, const int *position_id,
                         std::vector<uint16_t> &attention_mask,
                         int stage_idx = 0);
  int select_decode_stage();
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset = 0,
                  int size = 0);
  void init_by_names();
  int generate(bm_device_mem_t &logits_mem);
  int greedy_search(bm_device_mem_t &logits_mem);
  int penalty_sample(bm_device_mem_t &logits_mem);

public:
  int hidden_bytes;
  int kv_bytes;
  int token_length;
  int SEQLEN;
  int MAX_INPUT_LENGTH;
  int PREFILL_KV_LENGTH;
  int NUM_LAYERS;
  bool lmhead_with_topk;
  bool is_dynamic;
  bool prefill_mask;
  std::vector<int> visited_tokens;
  bool support_history;
  int history_length;
  uint16_t mask_value;
  bool is_same_addr;
  bool do_sample = false;

  // generation
  std::vector<int> eos_token_id;
  std::vector<std::string> stop_strings;
  float penalty;
  float temperature;
  int top_k;
  float top_p;
  int repetition_window; // sliding window for repetition penalty

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  std::vector<const bm_net_info_t *> net_blocks_kv;
  std::vector<int> decode_stage_len;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm, *net_greedy_head, *net_sample_head;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};
