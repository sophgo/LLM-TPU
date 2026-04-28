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

class EmbedVit {
public:
  void init(int devid, std::string model_path);
  void deinit();
  void forward_embed(ArrayInt const &tokens);
  void forward_vit(const float *pixel_values, ArrayInt const &position_ids,
                   ArrayInt const &pos_idx, ArrayFloat const &pos_weight,
                   ArrayInt const &grid_thw, int vit_offset);
  ArrayUint16 forward_embed_cache(int token);
  ArrayUint16 get_hidden_states();
  ArrayUint162D get_deepstacks();

  EmbedVit() {};

private:
  void init_by_names();

public:
  int token_length;
  int SEQLEN;
  int MAX_INPUT_LENGTH;
  int HIDDEN_SIZE;
  int VIT_DIMS;
  int MAX_PATCHES;
  int MAX_PIXELS;
  bool vit_dynamic;
  bool vit_run = false;
  int num_deepstack;
  std::vector<int> visited_tokens;

private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_vit;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> deepstack_buffers;
};