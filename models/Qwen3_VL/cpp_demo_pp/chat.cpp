//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "chat.hpp"
#include <algorithm>
#include <iostream>

void Qwen3_VL::init(std::vector<int> devids, std::string model_path,
                    std::string config_path, bool do_sample_) {
  // Auto-detect bmodel files in model_path directory by suffix pattern
  std::string embed_vit_path;
  std::string lmhead_path;
  std::vector<std::string> block_paths;

  for (auto &entry : std::filesystem::directory_iterator(model_path)) {
    auto filename = entry.path().filename().string();
    if (filename.size() < 7 || filename.substr(filename.size() - 7) != ".bmodel")
      continue;
    if (filename.find("embed_vit") != std::string::npos) {
      embed_vit_path = entry.path().string();
    } else if (filename.find("lmhead") != std::string::npos) {
      lmhead_path = entry.path().string();
    } else if (filename.find("block") != std::string::npos) {
      block_paths.push_back(entry.path().string());
    }
  }
  std::sort(block_paths.begin(), block_paths.end());

  if (embed_vit_path.empty()) {
    std::cerr << "Error: No embed_vit bmodel file found in " << model_path
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (lmhead_path.empty()) {
    std::cerr << "Error: No lmhead bmodel file found in " << model_path
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (block_paths.empty()) {
    std::cerr << "Error: No block bmodel files found in " << model_path
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // num_components: embed_vit + N blocks + lmhead
  int num_components = 1 + (int)block_paths.size() + 1;

  // Assign device IDs to components
  // Order: embed_vit, block_0, block_1, ..., block_n, lmhead
  std::vector<int> component_devids(num_components);
  if ((int)devids.size() == 1) {
    // All components on the same device
    std::fill(component_devids.begin(), component_devids.end(), devids[0]);
  } else if ((int)devids.size() >= num_components) {
    for (int i = 0; i < num_components; i++) {
      component_devids[i] = devids[i];
    }
  } else {
    // Distribute devices round-robin
    for (int i = 0; i < num_components; i++) {
      component_devids[i] = devids[i % devids.size()];
    }
  }

  printf("=== Multi-Device Configuration ===\n");
  printf("EmbedVit  -> Device %d [%s]\n", component_devids[0],
         embed_vit_path.c_str());
  for (int i = 0; i < (int)block_paths.size(); i++) {
    printf("Block[%d]  -> Device %d [%s]\n", i, component_devids[1 + i],
           block_paths[i].c_str());
  }
  printf("LmHead    -> Device %d [%s]\n", component_devids.back(),
         lmhead_path.c_str());
  printf("==================================\n");

  // Initialize EmbedVit
  embed_vit.init(component_devids[0], embed_vit_path);

  // Initialize Block instances
  blocks.resize(block_paths.size());
  for (int i = 0; i < (int)block_paths.size(); i++) {
    blocks[i].init(component_devids[1 + i], block_paths[i]);
  }

  // Initialize LmHead
  lmhead.init(component_devids.back(), lmhead_path, config_path, do_sample_);

  // Set public properties from components
  MAX_INPUT_LENGTH = embed_vit.MAX_INPUT_LENGTH;
  HIDDEN_SIZE = embed_vit.HIDDEN_SIZE;
  VIT_DIMS = embed_vit.VIT_DIMS;
  MAX_PATCHES = embed_vit.MAX_PATCHES;
  MAX_PIXELS = embed_vit.MAX_PIXELS;
  num_deepstack = embed_vit.num_deepstack;

  SEQLEN = blocks[0].SEQLEN;
  support_history = blocks[0].support_history;
  PREFILL_KV_LENGTH = blocks[0].PREFILL_KV_LENGTH;

  do_sample = do_sample_;
  history_length = 0;
  token_length = 0;

  visited_tokens.resize(SEQLEN);
  lmhead.visited_tokens.resize(SEQLEN);

  printf("Multi-device init done. Layers distributed across %d block(s).\n",
         (int)block_paths.size());
}

void Qwen3_VL::deinit() {
  embed_vit.deinit();
  for (auto &b : blocks) {
    b.deinit();
  }
  lmhead.deinit();
}

void Qwen3_VL::forward_embed(ArrayInt const &tokens) {
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());
  embed_vit.forward_embed(tokens);
  token_length = tokens.size();
}

void Qwen3_VL::forward_vit(const float *pixel_values,
                           ArrayInt const &position_ids,
                           ArrayInt const &pos_idx,
                           ArrayFloat const &pos_weight,
                           ArrayInt const &grid_thw, int vit_offset) {
  embed_vit.forward_vit(pixel_values, position_ids, pos_idx, pos_weight,
                        grid_thw, vit_offset);
}

int Qwen3_VL::forward_first(ArrayInt const &position_ids) {
  // D2S: transfer hidden states from EmbedVit device to host memory
  ArrayUint16 hidden_states = embed_vit.get_hidden_states();

  // D2S: transfer deepstack states from EmbedVit device to host memory
  ArrayUint162D all_deepstacks;
  if (embed_vit.vit_run) {
    all_deepstacks = embed_vit.get_deepstacks();
  }
  embed_vit.vit_run = false;

  int input_token_count = token_length;

  // Process through each Block instance
  for (int i = 0; i < (int)blocks.size(); i++) {
    // Sync state to block
    blocks[i].token_length = token_length;
    blocks[i].history_length = history_length;

    // Prepare deepstacks slice for this block's layers
    // Deepstacks correspond to global layer indices.
    // Only blocks with the "add" network can process deepstacks.
    ArrayUint162D block_deepstacks;
    int global_start = blocks[i].start_idx;
    for (int j = 0; j < blocks[i].num_blocks; j++) {
      int global_idx = global_start + j;
      if (global_idx < (int)all_deepstacks.size()) {
        block_deepstacks.push_back(all_deepstacks[global_idx]);
      }
    }

    // S2D + compute + D2S: Block processes on its device
    // Returns full sequence hidden states [token_length * HIDDEN_SIZE]
    hidden_states =
        blocks[i].forward_first(position_ids, hidden_states, block_deepstacks);
  }

  // Extract last token's hidden state for LmHead
  ArrayUint16 last_hidden(hidden_states.end() - HIDDEN_SIZE,
                          hidden_states.end());

  // Sync visited_tokens to LmHead for penalty sampling
  lmhead.token_length = token_length;
  std::copy(visited_tokens.begin(), visited_tokens.begin() + token_length,
            lmhead.visited_tokens.begin());

  // S2D + compute: LmHead generates token on its device
  int token = lmhead.forward(last_hidden);

  // Update wrapper state
  visited_tokens[token_length] = token;
  token_length++;
  if (support_history) {
    history_length += input_token_count + 1;
  } else {
    history_length = token_length;
  }

  return token;
}

int Qwen3_VL::forward_next(ArrayInt const &position_ids) {
  // Get last token for embedding cache
  int last_token = visited_tokens[token_length - 1];

  // D2S: EmbedVit produces single token embedding on its device
  ArrayUint16 hidden_states = embed_vit.forward_embed_cache(last_token);

  // Process through each Block instance
  for (int i = 0; i < (int)blocks.size(); i++) {
    // Sync state to block
    blocks[i].history_length = history_length;

    // S2D + compute + D2S: Block processes on its device
    hidden_states = blocks[i].forward_next(position_ids, hidden_states);
  }

  // Sync visited_tokens to LmHead for penalty sampling
  lmhead.token_length = token_length;
  std::copy(visited_tokens.begin(), visited_tokens.begin() + token_length,
            lmhead.visited_tokens.begin());

  // S2D + compute: LmHead generates token on its device
  int token = lmhead.forward(hidden_states);

  // Update wrapper state
  visited_tokens[token_length] = token;
  token_length++;
  history_length++;

  return token;
}

void Qwen3_VL::clear_history() {
  for (auto &b : blocks) {
    b.clear_history();
  }
  history_length = 0;
}

bool Qwen3_VL::check_stop(const std::string &text) {
  return lmhead.check_stop(text);
}
