//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "support.hpp"

class LmHead {
public:
  void init(int devid, std::string model_path, std::string config_path = "",
            bool do_sample = false);
  void deinit();
  int forward(ArrayUint16 &hidden_states);
  bool check_stop(const std::string &text);

  std::mt19937 sgen;
  LmHead() : sgen(std::random_device()()) {};

private:
  void init_by_names();
  int generate(bm_device_mem_t &logits_mem);
  int greedy_search(bm_device_mem_t &logits_mem);
  int penalty_sample(bm_device_mem_t &logits_mem);

public:
  int token_length;
  int SEQLEN;
  int HIDDEN_SIZE;
  bool lmhead_with_topk;
  bool do_sample = false;
  std::vector<int> visited_tokens;
  std::vector<std::string> stop_strings;
  float penalty;
  float temperature;
  int top_k;
  float top_p;

private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  const bm_net_info_t *net_lm;
  const bm_net_info_t *net_greedy_head;
  const bm_net_info_t *net_sample_head;
  bm_device_mem_t dev_buffer;
};