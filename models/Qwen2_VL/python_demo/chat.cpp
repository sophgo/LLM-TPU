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
               int stage_idx = 0) {
  for (int i = 0; i < net->input_num; i++) {
    empty(bm_handle, net->stages[stage_idx].input_mems[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    empty(bm_handle, net->stages[stage_idx].output_mems[i]);
  }
}

class Qwen2VL {
public:
  void init(int devid, std::string model_path);
  void deinit();
  void forward_embed(ArrayInt const &tokens);
  void forward_vit(ArrayFloat const &pixel_values, ArrayInt const &position_ids,
                   ArrayFloat const &full_attn_mask, ArrayInt const &grid_thw,
                   int vit_offset);
  int forward_first(ArrayInt const &position_ids);
  int forward_next(ArrayInt const &position_ids);

  std::mt19937 sgen;
  Qwen2VL() : sgen(std::random_device()()){};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  void net_launch_decode(int idx, int kv_offset, bm_device_mem_t &input_mem,
                         const int *position_id,
                         std::vector<uint16_t> &attention_mask);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  void init_by_names();

public:
  int token_length;
  int SEQLEN; // read from bmodel
  int HIDDEN_SIZE;
  int NUM_LAYERS; // read from bmodel
  int VIT_DIMS;
  int MAX_PATCHES;
  int MAX_PIXELS;
  int max_pos;
  const int spatial_merge_size = 2;
  bool lmhead_with_topk;
  uint16_t mask_value;
  int kv_bytes;      // size (bytes) of one token's K/V slice
  bool is_same_addr; // block input_mems[0] == output_mems[0] (in-place compute)

private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm, *net_greedy_head, *net_sample_head;
  const bm_net_info_t *net_vit;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};

void Qwen2VL::net_launch(const bm_net_info_t *net, int stage_idx) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  for (int i = 0; i < net->input_num; i++) {
    bmrt_tensor_with_device(
        &in_tensors[i], net->stages[stage_idx].input_mems[i],
        net->input_dtypes[i], net->stages[stage_idx].input_shapes[i]);
  }

  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(
        &out_tensors[i], net->stages[stage_idx].output_mems[i],
        net->output_dtypes[i], net->stages[stage_idx].output_shapes[i]);
  }
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  // bm_thread_sync(bm_handle);
}

// Decode-path block launcher. Ports the proven Qwen3 HEAD chat.cpp pattern:
// - in0 (hidden) is bound directly to the previous layer's output (no d2d).
// - in1/in2 (position_ids/attention_mask) are filled once on layer 0 and then
//   aliased from layer 0's buffers on later layers (no per-layer d2d).
// - out1/out2 (new K/V) are bound straight into the KV-cache slot at
//   kv_offset, so the net writes the new K/V in place — the post-launch
//   bm_memcpy_d2d_byte K/V scatter is dropped.
void Qwen2VL::net_launch_decode(int idx, int kv_offset,
                                bm_device_mem_t &input_mem,
                                const int *position_id,
                                std::vector<uint16_t> &attention_mask) {
  auto &net = net_blocks_cache[idx];
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  // in0: hidden state (caller-supplied, points at previous layer's output)
  bmrt_tensor_with_device(&in_tensors[0], input_mem, net->input_dtypes[0],
                          net->stages[0].input_shapes[0]);
  // in1: position_ids, in2: attention_mask — only filled once on layer 0 and
  // reused across layers by aliasing layer 0's buffers
  auto &in1_mem = net_blocks_cache[idx]->stages[0].input_mems[1];
  auto &in2_mem = net_blocks_cache[idx]->stages[0].input_mems[2];
  if (idx == 0) {
    bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id);
    bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    bmrt_tensor_with_device(&in_tensors[1], in1_mem, net->input_dtypes[1],
                            net->stages[0].input_shapes[1]);
    bmrt_tensor_with_device(&in_tensors[2], in2_mem, net->input_dtypes[2],
                            net->stages[0].input_shapes[2]);
  } else {
    bmrt_tensor_with_device(
        &in_tensors[1], net_blocks_cache[0]->stages[0].input_mems[1],
        net->input_dtypes[1], net->stages[0].input_shapes[1]);
    bmrt_tensor_with_device(
        &in_tensors[2], net_blocks_cache[0]->stages[0].input_mems[2],
        net->input_dtypes[2], net->stages[0].input_shapes[2]);
  }
  // in3: past_key, in4: past_value (the persistent KV cache)
  auto &in3_mem = net_blocks_cache[idx]->stages[0].input_mems[3];
  auto &in4_mem = net_blocks_cache[idx]->stages[0].input_mems[4];
  bmrt_tensor_with_device(&in_tensors[3], in3_mem, net->input_dtypes[3],
                          net->stages[0].input_shapes[3]);
  bmrt_tensor_with_device(&in_tensors[4], in4_mem, net->input_dtypes[4],
                          net->stages[0].input_shapes[4]);

  // out0: hidden state (next layer's input)
  auto &out0_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
  bmrt_tensor_with_device(&out_tensors[0], out0_mem, net->output_dtypes[0],
                          net->stages[0].output_shapes[0]);
  // out1/out2: new token's K/V — bind directly into the KV cache slot so the
  // net writes the new K/V in place, avoiding a separate d2d scatter copy.
  auto k_mem = bm_mem_from_device(
      past_key[idx].u.device.device_addr + kv_offset, kv_bytes);
  auto v_mem = bm_mem_from_device(
      past_value[idx].u.device.device_addr + kv_offset, kv_bytes);
  bmrt_tensor_with_device(&out_tensors[1], k_mem, net->output_dtypes[1],
                          net->stages[0].output_shapes[1]);
  bmrt_tensor_with_device(&out_tensors[2], v_mem, net->output_dtypes[2],
                          net->stages[0].output_shapes[2]);

  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   in_tensors.size(), out_tensors.data(),
                                   out_tensors.size(), true, false);
  assert(ret);
}

void Qwen2VL::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void Qwen2VL::init_by_names() {
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
  net_vit = bmrt_get_network_info(p_bmrt, "vit");
  const char **net_names = nullptr;
  auto num_nets = bmrt_get_network_number(p_bmrt);
  bmrt_get_network_names(p_bmrt, &net_names);
  net_greedy_head = nullptr;
  auto num_blocks =
      num_nets - 4; // 4 nets are embed, lm_head, embedding_cache, vit
  if (is_exist("greedy_head", net_names, num_nets)) {
    net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
    num_blocks--; // greedy_head is not a block
  }
  net_sample_head = nullptr;
  if (is_exist("sample_head", net_names, num_nets)) {
    net_sample_head = bmrt_get_network_info(p_bmrt, "sample_head");
    num_blocks--; // sample_head is not a block
  }

  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1]; // real seqlen
  lmhead_with_topk = net_lm->stages[0].output_shapes[0].dims[1] == 1;

  NUM_LAYERS = num_blocks / 2; // 2 nets for each block, one for cache
  // net blocks
  for (int i = 0; i < num_blocks / 2; i++) {
    auto block_name = "block_" + std::to_string(i);
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
}

void Qwen2VL::init(int dev_id, std::string model_path) {

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
  printf("Done!\n");
  print_devmem_info(bm_handle);
  init_by_names();
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  MAX_PATCHES = net_vit->stages[0].input_shapes[0].dims[0];

  VIT_DIMS = net_vit->stages[0].input_shapes[0].dims[1];
  MAX_PIXELS = MAX_PATCHES * 14 * 14;

  printf("Num Layers:%d\n", NUM_LAYERS);
  printf("Max Pixels: %d*%d*%d\n", MAX_PATCHES / 4, 28, 28);

  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  // size of one token's K (== V) slice; used to bind decode outputs in place
  kv_bytes = bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  // same addr
  is_same_addr = false;
  if (net_blocks[0]->stages[0].input_mems[0].u.device.device_addr ==
      net_blocks[0]->stages[0].output_mems[0].u.device.device_addr) {
    is_same_addr = true;
  }
  auto buffer_size =
      bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  assert(BM_SUCCESS == status);
}

void Qwen2VL::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

void Qwen2VL::forward_embed(ArrayInt const &tokens) {
  std::vector<int> input_ids(SEQLEN, 0);
  auto num = tokens.size();
  auto p_buffer = tokens.request();
  auto p_tokens = static_cast<int *>(p_buffer.ptr);
  std::copy(p_tokens, p_tokens + num, input_ids.data());

  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)input_ids.data());
  net_launch(net_embed);
  d2d(dev_buffer, out_mem);
  token_length = tokens.size();
}

void Qwen2VL::forward_vit(ArrayFloat const &pixel_values,
                          ArrayInt const &position_ids,
                          ArrayFloat const &full_attn_mask,
                          ArrayInt const &grid_thw, int vit_offset) {
  auto p_grid_thw = grid_thw.request();
  auto p_thw = static_cast<int *>(p_grid_thw.ptr);
  int t = p_thw[0];
  int h = p_thw[1];
  int w = p_thw[2];
  int hw = t * h * w;
  assert(full_attn_mask.size() == (hw * hw));
  assert(pixel_values.size() == (hw * VIT_DIMS));
  assert(position_ids.size() == (hw * 2));
  auto p_pixel_values = pixel_values.request();
  auto p_position_ids = position_ids.request();
  auto p_full_attn_mask = full_attn_mask.request();
  auto p_full = static_cast<float *>(p_full_attn_mask.ptr);
  empty_net(bm_handle, net_vit);
  auto &vit_in0_mem = net_vit->stages[0].input_mems[0];
  auto &vit_in1_mem = net_vit->stages[0].input_mems[1];
  auto &vit_in2_mem = net_vit->stages[0].input_mems[2];
  auto &vit_out_mem = net_vit->stages[0].output_mems[0];
  bm_memcpy_s2d_partial(bm_handle, vit_in0_mem, (void *)p_pixel_values.ptr,
                        pixel_values.size() * sizeof(float));
  bm_memcpy_s2d_partial(bm_handle, vit_in1_mem, (void *)p_position_ids.ptr,
                        position_ids.size() * sizeof(int));
  if (full_attn_mask.size() == MAX_PATCHES * MAX_PATCHES) {
    bm_memcpy_s2d(bm_handle, vit_in2_mem, (void *)p_full);
  } else {
    std::vector<float> mask_full(MAX_PATCHES * MAX_PATCHES, -10000.0f);
    for (int i = 0; i < hw; i++) {
      int mask_offset = i * MAX_PATCHES;
      int ori_offset = i * hw;
      std::copy(p_full + ori_offset, p_full + ori_offset + hw,
                mask_full.begin() + mask_offset);
    }
    bm_memcpy_s2d(bm_handle, vit_in2_mem, (void *)mask_full.data());
  }
  // launch vit
  net_launch(net_vit);

  // concatenante texting embedding and image embedding
  int dst_offset = vit_offset * HIDDEN_SIZE * sizeof(uint16_t);
  int vit_size = hw / 4 * HIDDEN_SIZE * sizeof(uint16_t);
  bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset, vit_out_mem, 0,
                     vit_size);
}

void Qwen2VL::head_launch(const bm_net_info_t *net,
                          bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  bmrt_tensor_with_device(&in_tensors[0], logits_mem, net->input_dtypes[0],
                          net->stages[0].input_shapes[0]);

  for (int i = 1; i < net->input_num; i++) {
    bmrt_tensor_with_device(&in_tensors[i], net->stages[0].input_mems[i],
                            net->input_dtypes[i],
                            net->stages[0].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(&out_tensors[i], net->stages[0].output_mems[i],
                            net->output_dtypes[i],
                            net->stages[0].output_shapes[i]);
  }
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  // bm_thread_sync(bm_handle);
}

int Qwen2VL::greedy_search(const bm_net_info_t *net,
                           bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];
  head_launch(net, logits_mem);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

int Qwen2VL::forward_first(ArrayInt const &position_ids) {
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, mask_value);
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < token_length; j++) {
      if (j <= i) {
        attention_mask[i * SEQLEN + j] = 0;
      }
    }
  }
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  std::vector<int> position_ids_pad(3 * SEQLEN, 0);
  int ori_length = position_ids.size() / 3;
  for (int i = 0; i < 3; i++) {
    int ori_offset = i * ori_length;
    int dst_offset = i * SEQLEN;
    std::copy(p_ids + ori_offset, p_ids + ori_offset + ori_length,
              position_ids_pad.begin() + dst_offset);
  }
  auto out_mem = dev_buffer;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    // d2d(in0_mem, block_out_mem);
    // In-place compute (input_mems[0] == output_mems[0]): once dev_buffer has
    // been copied in at idx==0, every later layer reads/writes the same shared
    // buffer, so the in0 copy is redundant. Mirrors Qwen3 HEAD chat.cpp.
    if (!is_same_addr || idx == 0) {
      d2d(in0_mem, out_mem);
    }
    if (idx == 0) {
      // only first time need copy
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_ids_pad.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    }
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2]);
  }

  // forward lmhead
  int bytes = HIDDEN_SIZE * sizeof(uint16_t);
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (token_length - 1) * bytes, bytes);
  net_launch(net_lm);
  int token = 0;
  if (lmhead_with_topk) {
    bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  } else {
    token = greedy_search(net_greedy_head, lm_out_mem);
  }
  token_length++;
  return token;
}

int Qwen2VL::forward_next(ArrayInt const &position_ids) {
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = mask_value;
  }
  assert(position_ids.size() == 3);
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  // embedding
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];

  auto in_mem = net_embed_cache->stages[0].input_mems[0];
  auto out_mem = net_embed_cache->stages[0].output_mems[0];
  if (lmhead_with_topk) {
    d2d(in_mem, lm_out_mem);
  } else {
    auto &greedy_out_mem = net_greedy_head->stages[0].output_mems[0];
    d2d(in_mem, greedy_out_mem);
  }
  net_launch(net_embed_cache);

  // blocks
  int token_offset = (token_length - 1) * kv_bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    // net_launch_decode feeds in0 (hidden) by aliasing the previous layer's
    // output, reuses layer 0's position_id/attention_mask buffers, and binds
    // out1/out2 (new K/V) directly into the KV cache slot — so the launch
    // writes the new K/V in place. No in0 d2d, no in1/in2 d2d, and no
    // post-launch K/V scatter copy are needed.
    net_launch_decode(idx, token_offset, out_mem, p_ids, attention_mask);
    out_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
  }

  // forward lmhead
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm);

  int token = 0;
  if (lmhead_with_topk) {
    bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  } else {
    token = greedy_search(net_greedy_head, lm_out_mem);
  }

  token_length++;
  return token;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Qwen2VL>(m, "Qwen2VL")
      .def(pybind11::init<>())
      .def("init", &Qwen2VL::init)
      .def("forward_embed", &Qwen2VL::forward_embed)
      .def("forward_vit", &Qwen2VL::forward_vit)
      .def("forward_first", &Qwen2VL::forward_first)
      .def("forward_next", &Qwen2VL::forward_next)
      .def("deinit", &Qwen2VL::deinit)
      .def_readonly("SEQLEN", &Qwen2VL::SEQLEN) // read SEQLEN in pipeline.py
      .def_readonly("MAX_PIXELS", &Qwen2VL::MAX_PIXELS)
      .def_readonly("MAX_PATCHES", &Qwen2VL::MAX_PATCHES)
      .def_readwrite("token_length", &Qwen2VL::token_length);
}
