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

class Qwen3_VL {
public:
  void init(int devid, std::string model_path, std::string config_path = "",
            bool do_sample = false, bool in_device = false);
  void deinit();
  void forward_embed(ArrayInt const &tokens);
  void forward_vit(const float *pixel_values, ArrayInt const &position_ids,
                   ArrayInt const &pos_idx, ArrayFloat const &pos_weight,
                   ArrayInt const &grid_thw, int vit_offset);
  int forward_first(ArrayInt const &position_ids);
  int forward_next(ArrayInt const &position_ids);
  bool check_stop(const std::string &text);
  void clear_history();

  std::mt19937 sgen;
  Qwen3_VL() : sgen(std::random_device()()) {};

private:
  void net_launch(const bm_net_info_t *net,
                  const std::vector<bm_tensor_t> &in_tensors,
                  std::vector<bm_tensor_t> &out_tensors);
  void net_launch_decode(int block_idx, int kv_offset,
                         bm_device_mem_t &input_mem, const int *position_id,
                         std::vector<uint16_t> &attention_mask);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset = 0,
                  int size = 0);
  void init_by_names();
  int forward_first_with_kv(ArrayInt const &position_ids);
  int generate(bm_device_mem_t &logits_mem);
  int greedy_search(bm_device_mem_t &logits_mem);
  int penalty_sample(bm_device_mem_t &logits_mem);
  void init_tensors(const bm_net_info_t *net,
                    std::vector<bm_tensor_t> &in_tensors,
                    std::vector<bm_tensor_t> &out_tensors, int stage = 0);

public:
  int token_length;
  int history_length;
  int SEQLEN;
  int MAX_INPUT_LENGTH;
  int PREFILL_KV_LENGTH;
  int HIDDEN_SIZE;
  int KV_BYTES; // kv bytes for one token
  int NUM_LAYERS;
  int VIT_DIMS;
  int MAX_PATCHES;
  int MAX_PIXELS;
  int max_pos;
  bool lmhead_with_topk;
  bool support_history;
  bool is_dynamic;
  bool vit_dynamic;
  uint16_t mask_value;
  bool vit_run = false;
  int num_deepstack;
  std::vector<int> visited_tokens;
  std::vector<int> VIT_PATCH_LIST;
  std::vector<int> INPUT_LENGTH_LIST;
  bool do_sample = false;
  // load bmodel to device memory
  bool in_device = false;
  bm_device_mem_u64_t bmodel_mem;
  // generation
  std::vector<std::string> stop_strings;
  float penalty;
  float temperature;
  int top_k;
  float top_p;

private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm;
  const bm_net_info_t *net_vit;
  const bm_net_info_t *net_add;
  const bm_net_info_t *net_greedy_head, *net_sample_head;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> deepstack_buffers;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};
