//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "block.hpp"
#include "embed_vit.hpp"
#include "lmhead.hpp"
#include <filesystem>

class Qwen3_VL {
public:
  void init(std::vector<int> devids, std::string model_path,
            std::string config_path = "", bool do_sample = false);
  void deinit();
  void forward_embed(ArrayInt const &tokens);
  void forward_vit(const float *pixel_values, ArrayInt const &position_ids,
                   ArrayInt const &pos_idx, ArrayFloat const &pos_weight,
                   ArrayInt const &grid_thw, int vit_offset);
  int forward_first(ArrayInt const &position_ids);
  int forward_next(ArrayInt const &position_ids);
  bool check_stop(const std::string &text);
  void clear_history();

  Qwen3_VL() {};

public:
  int token_length;
  int history_length;
  int SEQLEN;
  int MAX_INPUT_LENGTH;
  int PREFILL_KV_LENGTH;
  int HIDDEN_SIZE;
  int VIT_DIMS;
  int MAX_PATCHES;
  int MAX_PIXELS;
  bool support_history;
  bool do_sample;

private:
  EmbedVit embed_vit;
  std::vector<Block> blocks;
  LmHead lmhead;
  int num_deepstack;
  std::vector<int> visited_tokens;
};
