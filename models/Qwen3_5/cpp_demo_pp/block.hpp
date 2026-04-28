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

class Block {
public:
  void init(int devid, std::string model_path);
  void deinit();
  ArrayUint16 forward_first(ArrayInt const &position_ids,
                            ArrayUint16 &hidden_states);
  ArrayUint16 forward_next(ArrayInt const &position_ids,
                           ArrayUint16 &hidden_states);
  void clear_history();

  Block() {};

  // Return true if the layer at the given global index uses Full Attention
  // (and therefore has KV cache). Layers in-between use linear/recurrent
  // states. Pattern follows Qwen3_5: every FA_INTERVAL-th layer is FA.
  inline bool is_FA(int global_idx) const {
    return (global_idx + 1) % FA_INTERVAL == 0;
  }

private:
  void init_by_names();
  void net_launch_decode(int local_idx, int kv_offset, const int *position_id,
                         std::vector<uint16_t> &attention_mask);
  ArrayUint16 forward_first_with_kv(ArrayInt const &position_ids,
                                    ArrayUint16 &hidden_states);

public:
  int token_length;
  int history_length;
  int SEQLEN;
  int MAX_INPUT_LENGTH;
  int PREFILL_KV_LENGTH;
  int HIDDEN_SIZE;
  int KV_BYTES;
  int num_blocks; // number of layers in this Block instance
  int start_idx;  // global index of the first layer in this Block instance
  bool support_history;
  bool is_dynamic;
  uint16_t mask_value;
  const int FA_INTERVAL = 4;

private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  // KV cache (or conv/recurrent state for non-FA layers); indexed by local idx.
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};
