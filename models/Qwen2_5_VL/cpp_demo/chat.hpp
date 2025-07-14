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
#include <getopt.h>
#include <inttypes.h>
#include <iostream>
#include <numeric>
#include <random>
#include <stdio.h>
#include <vector>

typedef std::vector<int> ArrayInt;
typedef std::vector<float> ArrayFloat;
typedef std::vector<std::vector<int>> ArrayInt2D;
typedef std::vector<std::vector<float>> ArrayFloat2D;
static const uint16_t ATTENTION_MASK = 0xC61C; // -9984 by bfloat16

class Qwen2_5VL {
public:
  void init(int devid, std::string model_path);
  void deinit();
  void forward_embed(ArrayInt const &tokens);
  void forward_vit(ArrayFloat const &pixel_values, ArrayInt const &position_ids,
                   ArrayFloat const &full_attn_mask,
                   ArrayFloat const &window_attn_mask, ArrayInt const &grid_thw,
                   ArrayInt const &reverse_indices, int vit_offset);
  int forward_first(ArrayInt const &position_ids);
  int forward_next(ArrayInt const &position_ids);

  std::mt19937 sgen;
  Qwen2_5VL() : sgen(std::random_device()()) {};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  void init_by_names();

public:
  int token_length;
  int SEQLEN; // read from bmodel
  int MAX_INPUT_LENGTH; // read from bmodel
  int HIDDEN_SIZE;
  int NUM_LAYERS; // read from bmodel
  int VIT_DIMS;
  int MAX_PATCHES;
  int MAX_PIXELS;
  int max_pos;
  bool lmhead_with_topk;

private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm;
  const bm_net_info_t *net_vit;
  const bm_net_info_t *net_greedy_head, *net_sample_head;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};
