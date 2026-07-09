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

class LocateAnything {
public:
  void init(int devid, std::string model_path);
  void deinit();
  void forward_embed(ArrayInt const &tokens);
  void forward_vit(ArrayFloat const &pixel_values,
                   ArrayInt const &merger_index,
                   ArrayFloat const &pos_emb,
                   ArrayFloat const &rope_cos,
                   ArrayFloat const &rope_sin,
                   int vit_offset);
  int forward_first(ArrayInt const &position_ids);
  int forward_next(ArrayInt const &position_ids);
  void clear_history();

  std::mt19937 sgen;
  LocateAnything() : sgen(std::random_device()()){};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  void net_launch_block_dyn(const bm_net_info_t *net, int real_len);
  void net_launch_decode(int block_idx, int kv_offset,
                         bm_device_mem_t &input_mem, const int *position_id,
                         std::vector<uint16_t> &attention_mask);
  void vit_launch_dyn(int real_patches);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  void init_by_names();
  int greedy_search(bm_device_mem_t &logits_mem);

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
  bool prefill_mask;
  bool vit_dynamic;
  uint16_t mask_value;

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

void LocateAnything::net_launch(const bm_net_info_t *net, int stage_idx) {
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

void LocateAnything::net_launch_block_dyn(const bm_net_info_t *net, int real_len) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  for (int i = 0; i < net->input_num; i++) {
    bmrt_tensor_with_device(&in_tensors[i], net->stages[0].input_mems[i],
                            net->input_dtypes[i],
                            net->stages[0].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(&out_tensors[i], net->stages[0].output_mems[i],
                            net->output_dtypes[i],
                            net->stages[0].output_shapes[i]);
  }
  in_tensors[0].shape.dims[1] = real_len;
  in_tensors[1].shape.dims[1] = real_len;
  if (prefill_mask) {
    in_tensors[2].shape.dims[2] = real_len;
    in_tensors[2].shape.dims[3] = real_len;
  }

  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  // bm_thread_sync(bm_handle);
}

void LocateAnything::vit_launch_dyn(int real_patches) {
  std::vector<bm_tensor_t> in_tensors(net_vit->input_num);
  std::vector<bm_tensor_t> out_tensors(net_vit->output_num);

  for (int i = 0; i < net_vit->input_num; i++) {
    bmrt_tensor_with_device(&in_tensors[i], net_vit->stages[0].input_mems[i],
                            net_vit->input_dtypes[i],
                            net_vit->stages[0].input_shapes[i]);
  }
  for (int i = 0; i < net_vit->output_num; i++) {
    bmrt_tensor_with_device(&out_tensors[i], net_vit->stages[0].output_mems[i],
                            net_vit->output_dtypes[i],
                            net_vit->stages[0].output_shapes[i]);
  }
  // MoonViT: 5 inputs — update first dim for all
  in_tensors[0].shape.dims[0] = real_patches; // pixel_values
  in_tensors[1].shape.dims[0] = real_patches; // merger_index
  in_tensors[2].shape.dims[0] = real_patches; // pos_emb
  in_tensors[3].shape.dims[0] = real_patches; // rope_cos
  in_tensors[4].shape.dims[0] = real_patches; // rope_sin
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net_vit->name, in_tensors.data(),
                                   net_vit->input_num, out_tensors.data(),
                                   net_vit->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

void LocateAnything::net_launch_decode(int idx, int kv_offset,
                                  bm_device_mem_t &input_mem, const int *pos_id,
                                  std::vector<uint16_t> &attention_mask) {
  auto &net = net_blocks_cache[idx];
  std::vector<bm_tensor_t> in_tensors(5);
  std::vector<bm_tensor_t> out_tensors(3);
  // auto &in0_mem = net_blocks_cache[idx]->stages[0].input_mems[0];
  auto &in1_mem = net_blocks_cache[idx]->stages[0].input_mems[1];
  auto &in2_mem = net_blocks_cache[idx]->stages[0].input_mems[2];
  auto &in3_mem = net_blocks_cache[idx]->stages[0].input_mems[3];
  auto &in4_mem = net_blocks_cache[idx]->stages[0].input_mems[4];
  auto &out0_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
  // ===== prepare input tensors =====
  bmrt_tensor_with_device(&in_tensors[0], input_mem, net->input_dtypes[0],
                          net->stages[0].input_shapes[0]);
  if (idx == 0) {
    bm_memcpy_s2d(bm_handle, in1_mem, (void *)pos_id);
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
  bmrt_tensor_with_device(&in_tensors[3], in3_mem, net->input_dtypes[3],
                          net->stages[0].input_shapes[3]);
  bmrt_tensor_with_device(&in_tensors[4], in4_mem, net->input_dtypes[4],
                          net->stages[0].input_shapes[4]);
  // ===== prepare output tensors =====
  bmrt_tensor_with_device(&out_tensors[0], out0_mem, net->output_dtypes[0],
                          net->stages[0].output_shapes[0]);
  auto k_mem = bm_mem_from_device(
      past_key[idx].u.device.device_addr + kv_offset, KV_BYTES);
  auto v_mem = bm_mem_from_device(
      past_value[idx].u.device.device_addr + kv_offset, KV_BYTES);
  bmrt_tensor_with_device(&out_tensors[1], k_mem, net->output_dtypes[1],
                          net->stages[0].output_shapes[1]);
  bmrt_tensor_with_device(&out_tensors[2], v_mem, net->output_dtypes[2],
                          net->stages[0].output_shapes[2]);
  // ===== launch =====
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   in_tensors.size(), out_tensors.data(),
                                   out_tensors.size(), true, false);
  assert(ret);
}

void LocateAnything::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void LocateAnything::clear_history() {
  if (!support_history) {
    return;
  }
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  history_length = 0;
}

void LocateAnything::init_by_names() {
  auto is_exist = [](const char *name, const char **names, int num) {
    for (int i = 0; i < num; i++) {
      if (strcmp(name, names[i]) == 0) {
        return true;
      }
    }
    return false;
  };
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_vit = bmrt_get_network_info(p_bmrt, "vit");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  const char **net_names = nullptr;
  auto num_nets = bmrt_get_network_number(p_bmrt);
  bmrt_get_network_names(p_bmrt, &net_names);
  net_greedy_head = nullptr;
  // Count fixed networks: embed, lm_head, vit (+ embedding_cache if present)
  int num_fixed = 3; // embed + lm_head + vit
  net_embed_cache = nullptr;
  if (is_exist("embedding_cache", net_names, num_nets)) {
    net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
    num_fixed++;
  }
  auto num_blocks = num_nets - num_fixed;
  if (is_exist("greedy_head", net_names, num_nets)) {
    net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
    num_blocks--;
  }
  net_sample_head = nullptr;
  if (is_exist("sample_head", net_names, num_nets)) {
    net_sample_head = bmrt_get_network_info(p_bmrt, "sample_head");
    num_blocks--;
  }
  // 2 nets for each block, one for cache
  NUM_LAYERS = num_blocks / 2;

  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
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
  // Detect attention mask dtype from block output (bfloat16 or float16)
  if (net_blocks[0]->output_dtypes[0] == BM_FLOAT16) {
    mask_value = 0xF0E2; // float16 -10000
  } else if (net_blocks[0]->output_dtypes[0] == BM_BFLOAT16) {
    mask_value = 0xC61C; // bfloat16 -9984
  } else {
    std::cerr << "\nError: Invalid attention dtype\n";
    throw std::runtime_error("Invalid attention dtype");
  }
  support_history = net_blocks[0]->input_num == 5; // with kv cache
  is_dynamic = net_blocks[0]->is_dynamic;
  prefill_mask = net_blocks[0]->input_num > 2; // with prefill attention mask
  vit_dynamic = net_vit->is_dynamic;
  history_length = 0;
  lmhead_with_topk = net_lm->stages[0].output_shapes[0].dims[1] == 1;
  MAX_INPUT_LENGTH = net_embed->stages[0].input_shapes[0].dims[1];
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  SEQLEN = net_blocks_cache[0]->stages[0].input_shapes[3].dims[1];
  MAX_PATCHES = net_vit->stages[0].input_shapes[0].dims[0];
  MAX_PIXELS = MAX_PATCHES * 14 * 14;
  VIT_DIMS = net_vit->stages[0].input_shapes[0].dims[1];
  KV_BYTES =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  printf("Num Layers:%d\n", NUM_LAYERS);
  printf("Max Patches: %d, VIT Dims: %d\n", MAX_PATCHES, VIT_DIMS);
  printf("Max Pixels: %d\n", MAX_PIXELS);
  PREFILL_KV_LENGTH = 0;
  if (support_history) {
    PREFILL_KV_LENGTH = net_blocks[0]->stages[0].input_shapes[3].dims[1];
    printf("History Support: True\n");
  } else {
    printf("History Support: False\n");
  }
}

void LocateAnything::init(int dev_id, std::string model_path) {

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

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  auto buffer_size =
      bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  assert(BM_SUCCESS == status);
}

void LocateAnything::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

int LocateAnything::greedy_search(bm_device_mem_t &logits_mem) {
  auto &in_mem = net_greedy_head->stages[0].input_mems[0];
  auto &out_mem = net_greedy_head->stages[0].output_mems[0];
  d2d(in_mem, logits_mem);
  net_launch(net_greedy_head);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

void LocateAnything::forward_embed(ArrayInt const &tokens) {
  std::vector<int> input_ids(MAX_INPUT_LENGTH, 0);
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

void LocateAnything::forward_vit(ArrayFloat const &pixel_values,
                                  ArrayInt const &merger_index,
                                  ArrayFloat const &pos_emb,
                                  ArrayFloat const &rope_cos,
                                  ArrayFloat const &rope_sin,
                                  int vit_offset) {
  // MoonViT: 5 inputs (pixel_values, merger_index, pos_emb, rope_cos, rope_sin)
  int real_patches = pixel_values.size() / VIT_DIMS;
  int merge_ratio = 4; // 2x2 merge
  assert(merger_index.size() == real_patches);

  auto p_pixel_values = pixel_values.request();
  auto p_merger_index = merger_index.request();
  auto p_pos_emb = pos_emb.request();
  auto p_rope_cos = rope_cos.request();
  auto p_rope_sin = rope_sin.request();

  empty_net(bm_handle, net_vit);
  auto &vit_in0_mem = net_vit->stages[0].input_mems[0]; // pixel_values
  auto &vit_in1_mem = net_vit->stages[0].input_mems[1]; // merger_index
  auto &vit_in2_mem = net_vit->stages[0].input_mems[2]; // pos_emb
  auto &vit_in3_mem = net_vit->stages[0].input_mems[3]; // rope_cos
  auto &vit_in4_mem = net_vit->stages[0].input_mems[4]; // rope_sin
  auto &vit_out_mem = net_vit->stages[0].output_mems[0];

  bm_memcpy_s2d_partial(bm_handle, vit_in0_mem, (void *)p_pixel_values.ptr,
                        pixel_values.size() * sizeof(float));
  bm_memcpy_s2d_partial(bm_handle, vit_in1_mem, (void *)p_merger_index.ptr,
                        merger_index.size() * sizeof(int));
  bm_memcpy_s2d_partial(bm_handle, vit_in2_mem, (void *)p_pos_emb.ptr,
                        pos_emb.size() * sizeof(float));
  bm_memcpy_s2d_partial(bm_handle, vit_in3_mem, (void *)p_rope_cos.ptr,
                        rope_cos.size() * sizeof(float));
  bm_memcpy_s2d_partial(bm_handle, vit_in4_mem, (void *)p_rope_sin.ptr,
                        rope_sin.size() * sizeof(float));

  if (vit_dynamic) {
    vit_launch_dyn(real_patches);
  } else {
    net_launch(net_vit);
  }

  // Copy ViT output to dev_buffer at vit_offset
  // Output shape: [num_merged_tokens, HIDDEN_SIZE]
  int merged_tokens = real_patches / merge_ratio;
  int dst_offset = vit_offset * HIDDEN_SIZE * sizeof(uint16_t);
  int vit_size = merged_tokens * HIDDEN_SIZE * sizeof(uint16_t);
  bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset, vit_out_mem, 0,
                     vit_size);
}

void LocateAnything::head_launch(const bm_net_info_t *net,
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

int LocateAnything::forward_first(ArrayInt const &position_ids) {
  if (support_history) {
    // This demo does not support multi-turn history (prefill block has no
    // past-KV input). Should not be reached for this model.
    std::cerr << "[error] multi-turn history is not supported in this demo"
              << std::endl;
    assert(false);
  }
  std::vector<uint16_t> attention_mask;
  if (prefill_mask) {
    if (is_dynamic) {
      attention_mask.assign(token_length * token_length, mask_value);
      for (int i = 0; i < token_length; i++) {
        for (int j = 0; j <= i; j++) {
          attention_mask[i * token_length + j] = 0;
        }
      }
    } else {
      attention_mask.assign(MAX_INPUT_LENGTH * MAX_INPUT_LENGTH, mask_value);
      for (int i = 0; i < token_length; i++) {
        for (int j = 0; j <= i; j++) {
          attention_mask[i * MAX_INPUT_LENGTH + j] = 0;
        }
      }
    }
  }
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  std::vector<int> position_ids_pad;
  if (is_dynamic) {
    position_ids_pad.assign(token_length, 0);
    assert((int)position_ids.size() == token_length);
    std::copy(p_ids, p_ids + token_length, position_ids_pad.begin());
  } else {
    position_ids_pad.assign(MAX_INPUT_LENGTH, 0);
    std::copy(p_ids, p_ids + token_length, position_ids_pad.begin());
  }
  auto out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0]);
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    bm_memcpy_d2d_byte(bm_handle, in0_mem, 0, out_mem, 0,
                       token_length * HIDDEN_SIZE * sizeof(uint16_t));
    if (is_dynamic) {
      if (idx == 0) {
        // only first time need copy
        bm_memcpy_s2d_partial(bm_handle, in1_mem,
                              (void *)position_ids_pad.data(),
                              token_length * sizeof(int));
        if (prefill_mask) {
          bm_memcpy_s2d_partial(bm_handle, in2_mem, (void *)attention_mask.data(),
                                token_length * token_length * sizeof(uint16_t));
        } 
      }
      net_launch_block_dyn(net_blocks[idx], token_length);
    } else {
      if (idx == 0) {
        // only first time need copy
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_ids_pad.data());
        if (prefill_mask) {
          bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
        } 
      }
      net_launch(net_blocks[idx]);
    }
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
    token = greedy_search(lm_out_mem);
  }
  token_length++;
  history_length = token_length;
  return token;
}

int LocateAnything::forward_next(ArrayInt const &position_ids) {
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = history_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = mask_value;
  }
  assert(position_ids.size() == 1);
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
    d2d(in_mem, net_greedy_head->stages[0].output_mems[0]);
  }
  net_launch(net_embed_cache);

  // blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (history_length - 1) * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
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
    token = greedy_search(lm_out_mem);
  }
  token_length++;
  history_length++;
  return token;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<LocateAnything>(m, "LocateAnything")
      .def(pybind11::init<>())
      .def("init", &LocateAnything::init)
      .def("forward_embed", &LocateAnything::forward_embed)
      .def("forward_vit", &LocateAnything::forward_vit)
      .def("forward_first", &LocateAnything::forward_first)
      .def("forward_next", &LocateAnything::forward_next)
      .def("clear_history", &LocateAnything::clear_history)
      .def("deinit", &LocateAnything::deinit)
      .def_readonly("SEQLEN", &LocateAnything::SEQLEN) // read SEQLEN in pipeline.py
      .def_readonly("MAX_PIXELS", &LocateAnything::MAX_PIXELS)
      .def_readonly("MAX_PATCHES", &LocateAnything::MAX_PATCHES)
      .def_readonly("MAX_INPUT_LENGTH", &LocateAnything::MAX_INPUT_LENGTH)
      .def_readonly("PREFILL_KV_LENGTH", &LocateAnything::PREFILL_KV_LENGTH)
      .def_readonly("support_history", &LocateAnything::support_history)
      .def_readonly("history_length", &LocateAnything::history_length);
}
