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
                            ArrayUint16 &hidden_states,
                            ArrayUint162D &deepstacks);
  ArrayUint16 forward_next(ArrayInt const &position_ids,
                           ArrayUint16 &hidden_states);
  void clear_history();

  Block() {};

private:
  void init_by_names();
  void net_launch_decode(int block_idx, int kv_offset, const int *position_id,
                         std::vector<uint16_t> &attention_mask);
  ArrayUint16 forward_first_with_kv(ArrayInt const &position_ids,
                                    ArrayUint16 &hidden_states,
                                    ArrayUint162D &deepstacks);

public:
  int token_length;
  int history_length;
  int SEQLEN;
  int MAX_INPUT_LENGTH;
  int PREFILL_KV_LENGTH;
  int HIDDEN_SIZE;
  int KV_BYTES;
  int NUM_LAYERS;
  int num_blocks;
  int start_idx;
  bool support_history;
  bool is_dynamic;
  uint16_t mask_value;
  std::vector<int> visited_tokens;

private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_add;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};