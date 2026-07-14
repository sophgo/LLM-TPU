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
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdio.h>
#include <vector>

static void print_devmem_info(bm_handle_t &bm_handle) {
  bm_dev_stat_t stat;
  auto ret = bm_get_stat(bm_handle, &stat);
  if (ret != BM_SUCCESS) {
    std::cerr << "Failed to get device status" << std::endl;
    return;
  }
  std::cout << "DevMem: " << stat.mem_used << "/" << stat.mem_total << " MB"
            << std::endl;
}

namespace py = pybind11;
using ArrayFloat =
    py::array_t<float, py::array::c_style | py::array::forcecast>;
using ArrayInt = py::array_t<int, py::array::c_style | py::array::forcecast>;

//===------------------------------------------------------------===//
// Empty Func
//===------------------------------------------------------------===//
void empty(bm_handle_t &bm_handle, bm_device_mem_t &mem) {
  int value = 0;
  auto ret = bm_memset_device_ext(bm_handle, &value, 1, mem);
  assert(BM_SUCCESS == ret);
}

void empty_net(bm_handle_t &bm_handle, const bm_net_info_t *net,
               int stage = 0) {
  for (int i = 0; i < net->input_num; i++) {
    empty(bm_handle, net->stages[stage].input_mems[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    empty(bm_handle, net->stages[stage].output_mems[i]);
  }
}

class MiniCPMV4_6 {
public:
  void init(int devid, std::string model_path);
  void deinit();
  void forward_embed(ArrayInt const &tokens);
  void forward_vit(ArrayFloat const &pixel_values, ArrayInt const &pos_ids,
                   ArrayInt const &reorder_index,
                   ArrayInt const &window_index,
                   ArrayInt const &reverse_index,
                   std::string const &mode, int vit_offset);
  int forward_first(ArrayInt const &position_ids);
  int forward_next(ArrayInt const &position_ids);
  void clear_history();

  std::mt19937 sgen;
  MiniCPMV4_6() : sgen(std::random_device()()) {};

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
  inline bool is_FA(int layer_idx) {
    return (layer_idx + 1) % FA_INTERVAL == 0;
  }

public:
  int token_length;
  int history_length;
  int SEQLEN;
  int MAX_INPUT_LENGTH;
  int PREFILL_KV_LENGTH;
  int HIDDEN_SIZE;
  int KV_BYTES; // kv bytes for one token
  int NUM_LAYERS;
  int MAX_PATCHES;
  int MAX_PIXELS;
  int max_pos;
  bool lmhead_with_topk;
  bool support_history = false;
  bool prefill_mask = false;
  uint16_t mask_value;
  std::vector<int> visited_tokens;
  const int FA_INTERVAL = 4; // full attention interval

private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  std::vector<const bm_net_info_t *>
      net_blocks_prompt; // first time for full attention
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm;
  const bm_net_info_t *net_vit_4x;  // may be nullptr
  const bm_net_info_t *net_vit_16x; // may be nullptr
  const bm_net_info_t *net_greedy_head, *net_sample_head;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> past_key;   // conv state or past key
  std::vector<bm_device_mem_t> past_value; // recurrent state or past value
};

void MiniCPMV4_6::init_tensors(const bm_net_info_t *net,
                               std::vector<bm_tensor_t> &in_tensors,
                               std::vector<bm_tensor_t> &out_tensors,
                               int stage) {
  in_tensors.resize(net->input_num);
  out_tensors.resize(net->output_num);
  for (int i = 0; i < net->input_num; i++) {
    bmrt_tensor_with_device(&in_tensors[i], net->stages[stage].input_mems[i],
                            net->input_dtypes[i],
                            net->stages[stage].input_shapes[i]);
  }

  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(&out_tensors[i], net->stages[stage].output_mems[i],
                            net->output_dtypes[i],
                            net->stages[stage].output_shapes[i]);
  }
}

void MiniCPMV4_6::net_launch(const bm_net_info_t *net,
                             const std::vector<bm_tensor_t> &in_tensors,
                             std::vector<bm_tensor_t> &out_tensors) {
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
}

void MiniCPMV4_6::net_launch_decode(int idx, int kv_offset,
                                    bm_device_mem_t &input_mem,
                                    const int *pos_id,
                                    std::vector<uint16_t> &attention_mask) {
  auto &net = net_blocks_cache[idx];
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net, in_tensors, out_tensors);

  // ===== prepare input tensors =====
  in_tensors[0].device_mem = input_mem;
  bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem, (void *)pos_id);
  bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                (void *)attention_mask.data());
  out_tensors[1].device_mem = bm_mem_from_device(
      past_key[idx].u.device.device_addr + kv_offset, KV_BYTES);
  out_tensors[2].device_mem = bm_mem_from_device(
      past_value[idx].u.device.device_addr + kv_offset, KV_BYTES);

  // ===== launch =====
  net_launch(net, in_tensors, out_tensors);
}

void MiniCPMV4_6::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
                  int size) {
  if (!size) {
    size = bm_mem_get_device_size(src);
  }
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void MiniCPMV4_6::clear_history() {
  if (!support_history) {
    return;
  }
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  history_length = 0;
}

void MiniCPMV4_6::init_by_names() {
  auto is_exist = [](const char *name, const char **names, int num) {
    for (int i = 0; i < num; i++) {
      if (strcmp(name, names[i]) == 0) {
        return true;
      }
    }
    return false;
  };
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  const char **net_names = nullptr;
  auto num_nets = bmrt_get_network_number(p_bmrt);
  bmrt_get_network_names(p_bmrt, &net_names);
  net_greedy_head = nullptr;

  // Count non-block networks: embed, embed_cache, lm_head + vit(s)
  int num_extra = 3; // embed, embedding_cache, lm_head

  net_vit_4x = nullptr;
  net_vit_16x = nullptr;
  if (is_exist("vit_4x", net_names, num_nets)) {
    net_vit_4x = bmrt_get_network_info(p_bmrt, "vit_4x");
    num_extra++;
  }
  if (is_exist("vit_16x", net_names, num_nets)) {
    net_vit_16x = bmrt_get_network_info(p_bmrt, "vit_16x");
    num_extra++;
  }
  if (!net_vit_4x && !net_vit_16x) {
    std::cerr << "Error: No vit_4x or vit_16x found in bmodel\n";
    throw std::runtime_error("ViT network not found");
  }

  if (is_exist("greedy_head", net_names, num_nets)) {
    net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
    num_extra++;
  }
  net_sample_head = nullptr;
  if (is_exist("sample_head", net_names, num_nets)) {
    net_sample_head = bmrt_get_network_info(p_bmrt, "sample_head");
    num_extra++;
  }

  // Detect history support via block_kv networks
  std::string kv_name = "block_kv_" + std::to_string(FA_INTERVAL - 1);
  if (is_exist(kv_name.c_str(), net_names, num_nets)) {
    support_history = true;
  } else {
    support_history = false;
  }

  // 2 nets for each block, one for cache; +1 prompt net per FA block if history
  auto num_blocks = num_nets - num_extra;
  if (support_history) {
    NUM_LAYERS = num_blocks / (2 + 1.0 / FA_INTERVAL);
  } else {
    NUM_LAYERS = num_blocks / 2;
  }

  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    // With history, full-attention layers use the history block (block_kv_);
    // linear-attention layers always use block_. Without history, all layers
    // use block_.
    std::string block_name = (is_FA(i) && support_history)
                                 ? ("block_kv_" + std::to_string(i))
                                 : ("block_" + std::to_string(i));
    auto cache_name = "block_cache_" + std::to_string(i);
    if ((!is_exist(block_name.c_str(), net_names, num_nets)) ||
        (!is_exist(cache_name.c_str(), net_names, num_nets))) {
      NUM_LAYERS = i;
      printf("Warning: Only %d blocks found, expected %d blocks.\n", NUM_LAYERS,
             num_blocks / 2);
      break;
    }
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
    if (is_FA(i) && support_history) {
      // The full-attention prompt block is the normal prefill block (block_).
      auto prompt_name = "block_" + std::to_string(i);
      net_blocks_prompt.emplace_back(
          bmrt_get_network_info(p_bmrt, prompt_name.c_str()));
    }
  }
  free(net_names);
  if (net_embed_cache->output_dtypes[0] == BM_FLOAT16) {
    mask_value = 0xF0E2; // float16
  } else if (net_embed_cache->output_dtypes[0] == BM_BFLOAT16) {
    mask_value = 0xC61C; // -9984 by bfloat16
  } else {
    std::cerr << "\nError: Invalid attention dtype\n";
    std::cerr << "Supported dtype are 'BM_FLOAT16' or 'BM_BFLOAT16'\n";
    throw std::runtime_error("Invalid attention dtype");
  }
  history_length = 0;
  lmhead_with_topk = net_lm->stages[0].output_shapes[0].dims[1] == 1;
  MAX_INPUT_LENGTH = net_embed->stages[0].input_shapes[0].dims[1];
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  SEQLEN = net_blocks_cache[FA_INTERVAL - 1]->stages[0].input_shapes[3].dims[1];

  // Read MAX_PATCHES from ViT network (NaViT format: [1, 3, 14, patches*14])
  auto vit_ref = net_vit_4x ? net_vit_4x : net_vit_16x;
  MAX_PATCHES = vit_ref->stages[0].input_shapes[0].dims[3] / 14;
  MAX_PIXELS = MAX_PATCHES * 14 * 14;
  printf("ViT: %s%s%s\n",
         net_vit_4x ? "vit_4x" : "",
         (net_vit_4x && net_vit_16x) ? " + " : "",
         net_vit_16x ? "vit_16x" : "");
  printf("Max Patches: %d (Max Pixels: %d)\n", MAX_PATCHES, MAX_PIXELS);

  KV_BYTES = bm_mem_get_device_size(
      net_blocks_cache[FA_INTERVAL - 1]->stages[0].output_mems[1]);
  // with prefill attention mask
  prefill_mask =
      net_blocks[FA_INTERVAL - 1]->input_num == (support_history ? 5 : 3);
  printf("Num Layers:%d\n", NUM_LAYERS);
  PREFILL_KV_LENGTH = 0;
  if (support_history) {
    PREFILL_KV_LENGTH =
        net_blocks[FA_INTERVAL - 1]->stages[0].input_shapes[3].dims[1];
    printf("History Support: True\n");
  } else {
    printf("History Support: False\n");
  }
}

void MiniCPMV4_6::init(int dev_id, std::string model_path) {

  // request bm_handle
  std::cout << "Device [ " << dev_id << " ] loading .....\n";
  bm_status_t status = bm_dev_request(&bm_handle, dev_id);
  assert(BM_SUCCESS == status);

  // create bmruntime
  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  bm_thread_sync(bm_handle);
  printf("Done!\n");
  print_devmem_info(bm_handle);

  init_by_names();

  visited_tokens.resize(SEQLEN);

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    if (is_FA(i)) {
      // only init kv cache for layers with kv
      past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
      past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
      empty(bm_handle, past_key[i]);
      empty(bm_handle, past_value[i]);
    } else {
      // reuse key as conv state
      past_key[i] = net_blocks_cache[i]->stages[0].input_mems[1];
      // reuse value as recurrent state
      past_value[i] = net_blocks_cache[i]->stages[0].input_mems[2];
    }
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  if (!support_history) {
    auto buffer_size =
        bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
    status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
    assert(BM_SUCCESS == status);
  } else {
    // for history, we need a big buffer to store long input
    auto buffer_size = SEQLEN * HIDDEN_SIZE * sizeof(uint16_t);
    status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
    assert(BM_SUCCESS == status);
  }
}

void MiniCPMV4_6::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

int MiniCPMV4_6::greedy_search(bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_greedy_head, in_tensors, out_tensors);
  in_tensors[0].device_mem = logits_mem;
  net_launch(net_greedy_head, in_tensors, out_tensors);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_tensors[0].device_mem);
  return token;
}

void MiniCPMV4_6::forward_embed(ArrayInt const &tokens) {
  token_length = tokens.size();
  assert (token_length <= SEQLEN);
  auto p_buffer = tokens.request();
  auto p_tokens = static_cast<int *>(p_buffer.ptr);
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  std::copy(p_tokens, p_tokens + token_length, visited_tokens.data());
  if (!support_history) {
    assert(token_length <= MAX_INPUT_LENGTH);
  }
  empty(bm_handle, dev_buffer);
  for (int i = 0; i < token_length; i += MAX_INPUT_LENGTH) {
    std::vector<bm_tensor_t> in_tensors;
    std::vector<bm_tensor_t> out_tensors;
    init_tensors(net_embed, in_tensors, out_tensors);
    int real_len = std::min(MAX_INPUT_LENGTH, token_length - i);
    if (real_len != MAX_INPUT_LENGTH) {
      empty(bm_handle, in_tensors[0].device_mem);
    }
    bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                          (void *)(visited_tokens.data() + i),
                          real_len * sizeof(int));
    net_launch(net_embed, in_tensors, out_tensors);
    int offset = i * HIDDEN_SIZE * sizeof(uint16_t);
    d2d(dev_buffer, out_tensors[0].device_mem, offset,
        real_len * HIDDEN_SIZE * sizeof(uint16_t));
  }
}

void MiniCPMV4_6::forward_vit(ArrayFloat const &pixel_values,
                               ArrayInt const &pos_ids,
                               ArrayInt const &reorder_index,
                               ArrayInt const &window_index,
                               ArrayInt const &reverse_index,
                               std::string const &mode, int vit_offset) {
  // Select ViT network based on mode
  const bm_net_info_t *net = nullptr;
  if (mode == "4x") {
    net = net_vit_4x;
    if (!net) {
      std::cerr << "Error: vit_4x not found in bmodel\n";
      throw std::runtime_error("vit_4x not available");
    }
  } else {
    net = net_vit_16x;
    if (!net) {
      std::cerr << "Error: vit_16x not found in bmodel\n";
      throw std::runtime_error("vit_16x not available");
    }
  }

  auto p_pixel = pixel_values.request();
  auto p_pos = pos_ids.request();
  auto p_reorder = reorder_index.request();
  int num_patches = pos_ids.size();
  int num_reorder = reorder_index.size();
  int num_pixels = pixel_values.size(); // total float values in NaViT tensor

  empty_net(bm_handle, net);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net, in_tensors, out_tensors);

  // Input 0: pixel_values [1, 3, 14, patches*14]
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)p_pixel.ptr, num_pixels * sizeof(float));

  // Input 1: pos_ids [patches]
  bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                        (void *)p_pos.ptr, num_patches * sizeof(int));

  // Input 2: reorder_index [post_vit_patches]
  bm_memcpy_s2d_partial(bm_handle, in_tensors[2].device_mem,
                        (void *)p_reorder.ptr, num_reorder * sizeof(int));

  // Input 3-4 (16x only): window_index, reverse_index
  if (mode == "16x") {
    auto p_win = window_index.request();
    auto p_rev = reverse_index.request();
    bm_memcpy_s2d_partial(bm_handle, in_tensors[3].device_mem,
                          (void *)p_win.ptr, num_patches * sizeof(int));
    bm_memcpy_s2d_partial(bm_handle, in_tensors[4].device_mem,
                          (void *)p_rev.ptr, num_patches * sizeof(int));
    in_tensors[3].shape.dims[0] = num_patches;
    in_tensors[4].shape.dims[0] = num_patches;
  }

  // Update dynamic shapes
  in_tensors[0].shape.dims[3] = num_patches * 14;
  in_tensors[1].shape.dims[0] = num_patches;
  in_tensors[2].shape.dims[0] = num_reorder;

  net_launch(net, in_tensors, out_tensors);

  // Copy output to dev_buffer at the correct offset
  int dst_offset = vit_offset * HIDDEN_SIZE * sizeof(uint16_t);
  int vit_size =
      out_tensors[0].shape.dims[0] * HIDDEN_SIZE * sizeof(uint16_t);
  bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset,
                     out_tensors[0].device_mem, 0, vit_size);
}

int MiniCPMV4_6::generate(bm_device_mem_t &logits_mem) {
  int token = 0;
  if (lmhead_with_topk) {
    bm_memcpy_d2s_partial(bm_handle, (void *)&token, logits_mem, sizeof(int));
  } else {
    token = greedy_search(logits_mem);
  }
  return token;
}

int MiniCPMV4_6::forward_first(ArrayInt const &position_ids) {
  if (support_history) {
    return forward_first_with_kv(position_ids);
  }
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  std::vector<int> position_ids_pad;
  std::vector<uint16_t> attention_mask;
  if (prefill_mask) {
    attention_mask.resize(token_length * token_length, mask_value);
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * token_length + j] = 0;
      }
    }
  }
  position_ids_pad.assign(3 * token_length, 0);
  assert((int)position_ids.size() == token_length * 3);
  std::copy(p_ids, p_ids + token_length * 3, position_ids_pad.begin());

  auto out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0]);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    init_tensors(net_blocks[idx], in_tensors, out_tensors);
    out_tensors[0].device_mem = out_mem;
    d2d(in_tensors[0].device_mem, out_mem, 0,
        token_length * HIDDEN_SIZE * sizeof(uint16_t));
    if (is_FA(idx)) {
      bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                            (void *)position_ids_pad.data(),
                            token_length * 3 * sizeof(int));
      if (prefill_mask) {
        bm_memcpy_s2d_partial(bm_handle, in_tensors[2].device_mem,
                              (void *)attention_mask.data(),
                              token_length * token_length * sizeof(uint16_t));
        in_tensors[2].shape.dims[2] = token_length;
        in_tensors[2].shape.dims[3] = token_length;
      }
      in_tensors[0].shape.dims[1] = token_length;
      in_tensors[1].shape.dims[1] = token_length;
    } else {
      in_tensors[0].shape.dims[1] = token_length;
      empty(bm_handle, in_tensors[1].device_mem); // recurrent state
    }

    net_launch(net_blocks[idx], in_tensors, out_tensors);
    if (is_FA(idx)) {
      bm_memcpy_d2d_byte(bm_handle, past_key[idx], 0,
                         net_blocks[idx]->stages[0].output_mems[1], 0,
                         KV_BYTES * token_length);
      bm_memcpy_d2d_byte(bm_handle, past_value[idx], 0,
                         net_blocks[idx]->stages[0].output_mems[2], 0,
                         KV_BYTES * token_length);
    } else {
      // reuse key as conv state
      d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
      // reuse value as recurrent state
      d2d(past_value[idx], net_blocks[idx]->stages[0].input_mems[1]);
    }
  }

  // forward lmhead
  int bytes = HIDDEN_SIZE * sizeof(uint16_t);
  init_tensors(net_lm, in_tensors, out_tensors);
  in_tensors[0].device_mem = bm_mem_from_device(
      out_mem.u.device.device_addr + (token_length - 1) * bytes, bytes);
  out_tensors[0].device_mem = dev_buffer;
  net_launch(net_lm, in_tensors, out_tensors);
  auto token = generate(dev_buffer);
  visited_tokens[token_length] = token;
  token_length++;
  history_length = token_length;
  return token;
}

int MiniCPMV4_6::forward_first_with_kv(ArrayInt const &position_ids) {
  assert(history_length + token_length < SEQLEN);
  assert(prefill_mask == false);
  assert((int)position_ids.size() == 3 * token_length);
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  std::vector<int> pos_ids(3 * MAX_INPUT_LENGTH, 0);

  auto out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0]);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  int k_idx = prefill_mask ? 3 : 2;
  int old_kvlen = (history_length > 0) ? (history_length - 1) : 0;
  int last_cur_len = 0; // cur_len of the last chunk, for lmhead offset
  for (int t = 0; t < token_length; t += MAX_INPUT_LENGTH) {
    auto old_length = history_length;
    int cur_len = std::min(MAX_INPUT_LENGTH, token_length - t);
    last_cur_len = cur_len;
    history_length += cur_len;
    // copy position ids with offset
    for (int i = 0; i < 3; i++) {
      std::copy(p_ids + i * token_length + t,
                p_ids + i * token_length + t + cur_len,
                pos_ids.data() + i * cur_len);
    }

    assert(old_length <= PREFILL_KV_LENGTH);
    for (int idx = 0; idx < NUM_LAYERS; idx++) {
      auto &net = (is_FA(idx) && old_length == 0 && t == 0)
                      ? net_blocks_prompt[idx / FA_INTERVAL]
                      : net_blocks[idx];
      init_tensors(net, in_tensors, out_tensors);
      out_tensors[0].device_mem = out_mem;

      if (idx == 0) {
        // for the first block, copy input from dev_buffer with offset
        bm_memcpy_d2d_byte(bm_handle, in_tensors[0].device_mem, 0, dev_buffer,
                           t * HIDDEN_SIZE * sizeof(uint16_t),
                           cur_len * HIDDEN_SIZE * sizeof(uint16_t));
      } else {
        d2d(in_tensors[0].device_mem, out_mem, 0,
            cur_len * HIDDEN_SIZE * sizeof(uint16_t));
      }
      if (is_FA(idx)) {
        bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                              (void *)(pos_ids.data()),
                              cur_len * 3 * sizeof(int));
        in_tensors[0].shape.dims[1] = cur_len;
        in_tensors[1].shape.dims[1] = cur_len;
        // copy old kv to new kv with offset
        if (old_kvlen > 0) {
          d2d(in_tensors[k_idx].device_mem, past_key[idx], 0,
              KV_BYTES * old_kvlen);
          d2d(in_tensors[k_idx + 1].device_mem, past_value[idx], 0,
              KV_BYTES * old_kvlen);
          in_tensors[k_idx].shape.dims[1] = old_kvlen;
          in_tensors[k_idx + 1].shape.dims[1] = old_kvlen;
        }
      } else {
        if (old_kvlen > 0) {
          d2d(in_tensors[1].device_mem, past_value[idx]);
          d2d(in_tensors[2].device_mem, past_key[idx]);
        } else {
          empty(bm_handle, in_tensors[1].device_mem); // recurrent state
          empty(bm_handle, in_tensors[2].device_mem); // conv state
        }
        in_tensors[0].shape.dims[1] = cur_len;
      }

      net_launch(net, in_tensors, out_tensors);
      if (is_FA(idx)) {
        size_t offset = old_kvlen * KV_BYTES;
        bm_memcpy_d2d_byte(bm_handle, past_key[idx], offset,
                           net->stages[0].output_mems[1], 0,
                           KV_BYTES * cur_len);
        bm_memcpy_d2d_byte(bm_handle, past_value[idx], offset,
                           net->stages[0].output_mems[2], 0,
                           KV_BYTES * cur_len);
      } else {
        // reuse key as conv state
        d2d(past_key[idx], net->stages[0].output_mems[1]);
        // reuse value as recurrent state
        d2d(past_value[idx], net->stages[0].input_mems[1]);
      }
    }
    old_kvlen += cur_len;
  }
  // forward lmhead
  int bytes = HIDDEN_SIZE * sizeof(uint16_t);
  init_tensors(net_lm, in_tensors, out_tensors);
  in_tensors[0].device_mem = bm_mem_from_device(
      out_mem.u.device.device_addr + (last_cur_len - 1) * bytes, bytes);
  out_tensors[0].device_mem = dev_buffer;
  net_launch(net_lm, in_tensors, out_tensors);
  int token = generate(dev_buffer);
  visited_tokens[token_length] = token;
  token_length++;
  history_length++;
  return token;
}

int MiniCPMV4_6::forward_next(ArrayInt const &position_ids) {
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = history_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = mask_value;
  }
  assert(position_ids.size() == 3);
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  // embedding
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_embed_cache, in_tensors, out_tensors);
  int token = visited_tokens[token_length - 1];
  bm_memcpy_s2d(bm_handle, in_tensors[0].device_mem, (void *)&token);
  net_launch(net_embed_cache, in_tensors, out_tensors);
  auto out_mem = out_tensors[0].device_mem;

  // blocks
  int fa_bytes = bm_mem_get_device_size(
      net_blocks_cache[FA_INTERVAL - 1]->stages[0].output_mems[1]);
  int token_offset = (history_length - 1) * fa_bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    if (is_FA(idx)) {
      net_launch_decode(idx, token_offset, out_mem, p_ids, attention_mask);
    } else {
      init_tensors(net_blocks_cache[idx], in_tensors, out_tensors);
      in_tensors[0].device_mem = out_mem;
      net_launch(net_blocks_cache[idx], in_tensors, out_tensors);
    }
    out_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
  }

  // forward lmhead
  init_tensors(net_lm, in_tensors, out_tensors);
  in_tensors[0].device_mem = out_mem;
  out_tensors[0].device_mem = dev_buffer;
  net_launch(net_lm, in_tensors, out_tensors);

  token = generate(dev_buffer);
  visited_tokens[token_length] = token;
  token_length++;
  history_length++;
  return token;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<MiniCPMV4_6>(m, "MiniCPMV4_6")
      .def(pybind11::init<>())
      .def("init", &MiniCPMV4_6::init)
      .def("forward_embed", &MiniCPMV4_6::forward_embed)
      .def("forward_vit", &MiniCPMV4_6::forward_vit)
      .def("forward_first", &MiniCPMV4_6::forward_first)
      .def("forward_next", &MiniCPMV4_6::forward_next)
      .def("clear_history", &MiniCPMV4_6::clear_history)
      .def("deinit", &MiniCPMV4_6::deinit)
      .def_readonly("SEQLEN", &MiniCPMV4_6::SEQLEN)
      .def_readonly("MAX_PIXELS", &MiniCPMV4_6::MAX_PIXELS)
      .def_readonly("MAX_PATCHES", &MiniCPMV4_6::MAX_PATCHES)
      .def_readonly("MAX_INPUT_LENGTH", &MiniCPMV4_6::MAX_INPUT_LENGTH)
      .def_readonly("PREFILL_KV_LENGTH", &MiniCPMV4_6::PREFILL_KV_LENGTH)
      .def_readonly("support_history", &MiniCPMV4_6::support_history)
      .def_readonly("history_length", &MiniCPMV4_6::history_length);
}
