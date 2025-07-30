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

namespace py = pybind11;
using ArrayFloat =
    py::array_t<float, py::array::c_style | py::array::forcecast>;
using ArrayInt = py::array_t<int, py::array::c_style | py::array::forcecast>;
static const uint16_t ATTENTION_MASK = 0xC61C; // -9984 by bfloat16

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

class Qwen2_5O {
public:
  void init(int devid, std::string model_path);
  void deinit();
  void forward_embed(ArrayInt const &tokens);
  void forward_vit(ArrayFloat const &pixel_values, ArrayInt const &position_ids,
                   ArrayFloat const &full_attn_mask,
                   ArrayFloat const &window_attn_mask, ArrayInt const &grid_thw,
                   ArrayInt const &reverse_indices, int vit_offset,
                   bool use_video_buffer);
  void forward_audio(ArrayFloat const &audio_features,
                     ArrayInt const &audio_offset);
  void video_sync(int chunck_tokens, ArrayInt const &video_offset);
  int forward_first(ArrayInt const &position_ids);
  int forward_next(ArrayInt const &position_ids);
  void clear_history();

  std::mt19937 sgen;
  Qwen2_5O() : sgen(std::random_device()()) {};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  void net_launch_block_dyn(const bm_net_info_t *net, int real_len);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  void init_by_names();
  int forward_first_with_kv(ArrayInt const &position_ids);
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

private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed, *net_embed_cache, *net_lm;
  const bm_net_info_t *net_vit, *net_audio;
  const bm_net_info_t *net_greedy_head, *net_sample_head;
  bm_device_mem_t dev_buffer;
  bm_device_mem_t video_buffer;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};

void Qwen2_5O::net_launch(const bm_net_info_t *net, int stage_idx) {
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
  bm_thread_sync(bm_handle);
}

void Qwen2_5O::net_launch_block_dyn(const bm_net_info_t *net, int real_len) {
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
  in_tensors[2].shape.dims[2] = real_len;
  in_tensors[2].shape.dims[3] = real_len;

  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

void Qwen2_5O::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void Qwen2_5O::clear_history() {
  if (!support_history) {
    return;
  }
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  history_length = 0;
}

void Qwen2_5O::init_by_names() {
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
  net_vit = bmrt_get_network_info(p_bmrt, "vit");
  net_audio = bmrt_get_network_info(p_bmrt, "audio");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  const char **net_names = nullptr;
  auto num_nets = bmrt_get_network_number(p_bmrt);
  bmrt_get_network_names(p_bmrt, &net_names);
  net_greedy_head = nullptr;
  // 5 nets are embed, lm_head, embedding_cache, vit, audio
  auto num_blocks = num_nets - 5;
  if (is_exist("greedy_head", net_names, num_nets)) {
    net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
    num_blocks--; // greedy_head is not a block
  }
  net_sample_head = nullptr;
  if (is_exist("sample_head", net_names, num_nets)) {
    net_sample_head = bmrt_get_network_info(p_bmrt, "sample_head");
    num_blocks--; // sample_head is not a block
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
  support_history = net_blocks[0]->input_num == 5; // with kv cache
  is_dynamic = net_blocks[0]->is_dynamic;
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
  printf("Max Pixels: %d*%d*%d\n", MAX_PATCHES / 4, 28, 28);
  PREFILL_KV_LENGTH = 0;
  if (support_history) {
    PREFILL_KV_LENGTH = net_blocks[0]->stages[0].input_shapes[3].dims[1];
    printf("History Support: True\n");
  } else {
    printf("History Support: False\n");
  }
}

void Qwen2_5O::init(int dev_id, std::string model_path) {

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
  status = bm_malloc_device_byte(bm_handle, &video_buffer, buffer_size);
  assert(BM_SUCCESS == status);
}

void Qwen2_5O::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bm_free_device(bm_handle, video_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

int Qwen2_5O::greedy_search(bm_device_mem_t &logits_mem) {
  auto &in_mem = net_greedy_head->stages[0].input_mems[0];
  auto &out_mem = net_greedy_head->stages[0].output_mems[0];
  d2d(in_mem, logits_mem);
  net_launch(net_greedy_head);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

void Qwen2_5O::forward_embed(ArrayInt const &tokens) {
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

void Qwen2_5O::forward_vit(ArrayFloat const &pixel_values,
                           ArrayInt const &position_ids,
                           ArrayFloat const &full_attn_mask,
                           ArrayFloat const &window_attn_mask,
                           ArrayInt const &grid_thw,
                           ArrayInt const &reverse_indices, int vit_offset,
                           bool use_video_buffer) {
  auto p_grid_thw = grid_thw.request();
  auto p_thw = static_cast<int *>(p_grid_thw.ptr);
  int t = p_thw[0];
  int h = p_thw[1];
  int w = p_thw[2];
  int hw = t * h * w;
  assert(full_attn_mask.size() == (hw * hw));
  assert(window_attn_mask.size() == (hw * hw));
  assert(pixel_values.size() == (hw * VIT_DIMS));
  assert(position_ids.size() == (hw * 2));
  assert(reverse_indices.size() == (hw / 4));
  auto p_pixel_values = pixel_values.request();
  auto p_position_ids = position_ids.request();
  auto p_full_attn_mask = full_attn_mask.request();
  auto p_window_attn_mask = window_attn_mask.request();
  auto p_reverse_indices = reverse_indices.request();
  auto p_full = static_cast<float *>(p_full_attn_mask.ptr);
  auto p_window = static_cast<float *>(p_window_attn_mask.ptr);
  empty_net(bm_handle, net_vit);
  auto &vit_in0_mem = net_vit->stages[0].input_mems[0];
  auto &vit_in1_mem = net_vit->stages[0].input_mems[1];
  auto &vit_in2_mem = net_vit->stages[0].input_mems[2];
  auto &vit_in3_mem = net_vit->stages[0].input_mems[3];
  auto &vit_in4_mem = net_vit->stages[0].input_mems[4];
  auto &vit_out_mem = net_vit->stages[0].output_mems[0];
  bm_memcpy_s2d_partial(bm_handle, vit_in0_mem, (void *)p_pixel_values.ptr,
                        pixel_values.size() * sizeof(float));
  bm_memcpy_s2d_partial(bm_handle, vit_in1_mem, (void *)p_position_ids.ptr,
                        position_ids.size() * sizeof(int));
  bm_memcpy_s2d_partial(bm_handle, vit_in4_mem, (void *)p_reverse_indices.ptr,
                        reverse_indices.size() * sizeof(int));
  if (full_attn_mask.size() == MAX_PATCHES * MAX_PATCHES) {
    bm_memcpy_s2d(bm_handle, vit_in2_mem, (void *)p_full);
    bm_memcpy_s2d(bm_handle, vit_in3_mem, (void *)p_window);
  } else {
    std::vector<float> mask_full(MAX_PATCHES * MAX_PATCHES, -10000.0f);
    std::vector<float> mask_window(MAX_PATCHES * MAX_PATCHES, -10000.0f);

    for (int i = 0; i < hw; i++) {
      int mask_offset = i * MAX_PATCHES;
      int ori_offset = i * hw;
      std::copy(p_full + ori_offset, p_full + ori_offset + hw,
                mask_full.begin() + mask_offset);
      std::copy(p_window + ori_offset, p_window + ori_offset + hw,
                mask_window.begin() + mask_offset);
    }
    bm_memcpy_s2d(bm_handle, vit_in2_mem, (void *)mask_full.data());
    bm_memcpy_s2d(bm_handle, vit_in3_mem, (void *)mask_window.data());
  }
  // launch vit
  net_launch(net_vit);

  // concatenante texting embedding and image embedding
  int dst_offset = vit_offset * HIDDEN_SIZE * sizeof(uint16_t);
  int vit_size = hw / 4 * HIDDEN_SIZE * sizeof(uint16_t);
  auto buffer = use_video_buffer ? video_buffer : dev_buffer;
  bm_memcpy_d2d_byte(bm_handle, buffer, dst_offset, vit_out_mem, 0, vit_size);
}

void Qwen2_5O::video_sync(int chunck_tokens, ArrayInt const &video_offset) {
  auto p_offsets = video_offset.request();
  auto p_video_offsets = static_cast<int *>(p_offsets.ptr);
  int num_video = video_offset.size();
  int in_offset = 0;
  int bytes = chunck_tokens * HIDDEN_SIZE * sizeof(uint16_t);
  for (int i = 0; i < num_video; i++) {
    int dst_offset = p_video_offsets[i] * HIDDEN_SIZE * sizeof(uint16_t);
    bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset, video_buffer,
                       in_offset, bytes);
    in_offset += bytes;
  }
}

void Qwen2_5O::forward_audio(ArrayFloat const &audio_features,
                             ArrayInt const &audio_offset) {
  auto p_audio_features = audio_features.request();
  auto p_audio = static_cast<float *>(p_audio_features.ptr);
  auto p_audio_offset = audio_offset.request();
  auto p_offsets = static_cast<int *>(p_audio_offset.ptr);
  int num_audio = audio_offset.size();
  empty_net(bm_handle, net_audio);
  auto &audio_in_mem = net_audio->stages[0].input_mems[0];
  auto &audio_out_mem = net_audio->stages[0].output_mems[0];
  auto audio_in_bytes = bm_mem_get_device_size(audio_in_mem);
  auto audio_out_bytes = bm_mem_get_device_size(audio_out_mem);
  for (int i = 0; i < num_audio; i++) {
    int in_offset = i * audio_in_bytes / sizeof(float);
    bm_memcpy_s2d(bm_handle, audio_in_mem, (void *)(p_audio + in_offset));
    net_launch(net_audio);
    // copy to dev_buffer
    int dst_offset = p_offsets[i] * HIDDEN_SIZE * sizeof(uint16_t);
    bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset, audio_out_mem, 0,
                       audio_out_bytes);
  }
}

void Qwen2_5O::head_launch(const bm_net_info_t *net,
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
  bm_thread_sync(bm_handle);
}

int Qwen2_5O::forward_first(ArrayInt const &position_ids) {
  if (support_history) {
    return forward_first_with_kv(position_ids);
  }
  std::vector<uint16_t> attention_mask;
  if (is_dynamic) {
    attention_mask.assign(token_length * token_length, ATTENTION_MASK);
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * token_length + j] = 0;
      }
    }
  } else {
    attention_mask.assign(MAX_INPUT_LENGTH * MAX_INPUT_LENGTH, ATTENTION_MASK);
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * MAX_INPUT_LENGTH + j] = 0;
      }
    }
  }
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  std::vector<int> position_ids_pad;
  if (is_dynamic) {
    position_ids_pad.assign(3 * token_length, 0);
    assert((int)position_ids.size() == token_length * 3);
    std::copy(p_ids, p_ids + token_length * 3, position_ids_pad.begin());
  } else {
    position_ids_pad.assign(3 * MAX_INPUT_LENGTH, 0);
    int ori_length = position_ids.size() / 3;
    for (int i = 0; i < 3; i++) {
      int ori_offset = i * ori_length;
      int dst_offset = i * MAX_INPUT_LENGTH;
      std::copy(p_ids + ori_offset, p_ids + ori_offset + ori_length,
                position_ids_pad.begin() + dst_offset);
    }
  }
  auto out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0]);
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    d2d(in0_mem, out_mem);
    if (is_dynamic) {
      if (idx == 0) {
        // only first time need copy
        bm_memcpy_s2d_partial(bm_handle, in1_mem,
                              (void *)position_ids_pad.data(),
                              token_length * 3 * sizeof(int));
        bm_memcpy_s2d_partial(bm_handle, in2_mem, (void *)attention_mask.data(),
                              token_length * token_length * sizeof(uint16_t));
      }
      net_launch_block_dyn(net_blocks[idx], token_length);
    } else {
      if (idx == 0) {
        // only first time need copy
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_ids_pad.data());
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
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

int Qwen2_5O::forward_first_with_kv(ArrayInt const &position_ids) {
  int max_kv_length = MAX_INPUT_LENGTH + PREFILL_KV_LENGTH;
  std::vector<uint16_t> attention_mask(MAX_INPUT_LENGTH * max_kv_length,
                                       ATTENTION_MASK);
  auto old_length = history_length;
  history_length += token_length;
  assert(history_length < SEQLEN);
  assert(old_length <= PREFILL_KV_LENGTH);
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < old_length; j++) {
      attention_mask[i * max_kv_length + j] = 0;
    }
    for (int j = 0; j <= i; j++) {
      attention_mask[i * max_kv_length + j + PREFILL_KV_LENGTH] = 0;
    }
  }
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);

  std::vector<int> position_ids_pad(3 * MAX_INPUT_LENGTH, 0);
  int ori_length = position_ids.size() / 3;
  assert(ori_length == token_length);
  assert(ori_length <= MAX_INPUT_LENGTH);
  for (int i = 0; i < 3; i++) {
    int ori_offset = i * ori_length;
    int dst_offset = i * MAX_INPUT_LENGTH;
    std::copy(p_ids + ori_offset, p_ids + ori_offset + ori_length,
              position_ids_pad.begin() + dst_offset);
  }

  auto out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0]);
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    auto &in3_mem = net_blocks[idx]->stages[0].input_mems[3];
    auto &in4_mem = net_blocks[idx]->stages[0].input_mems[4];

    d2d(in0_mem, out_mem);
    if (old_length > 0) {
      bm_memcpy_d2d_byte(bm_handle, in3_mem, 0, past_key[idx], 0,
                         KV_BYTES * old_length);
      bm_memcpy_d2d_byte(bm_handle, in4_mem, 0, past_value[idx], 0,
                         KV_BYTES * old_length);
    } else if (idx == 0) {
      empty(bm_handle, in3_mem);
      empty(bm_handle, in4_mem);
    }
    bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_ids_pad.data());
    bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    auto &out1_mem = net_blocks[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks[idx]->stages[0].output_mems[2];
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], old_length * KV_BYTES,
                       out1_mem, 0, KV_BYTES * token_length);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], old_length * KV_BYTES,
                       out2_mem, 0, KV_BYTES * token_length);
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
  history_length++;
  return token;
}

int Qwen2_5O::forward_next(ArrayInt const &position_ids) {
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = history_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = ATTENTION_MASK;
  }
  assert(position_ids.size() == 3);
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  // embedding
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];

  auto &in_mem = net_embed_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_cache->stages[0].output_mems[0];
  d2d(in_mem, lm_out_mem);
  net_launch(net_embed_cache);

  // blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (history_length - 1) * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks_cache[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks_cache[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks_cache[idx]->stages[0].input_mems[2];
    auto &out0_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
    auto &out1_mem = net_blocks_cache[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks_cache[idx]->stages[0].output_mems[2];
    d2d(in0_mem, out_mem);
    if (idx == 0) {
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)p_ids);
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    } else {
      d2d(in1_mem, net_blocks_cache[0]->stages[0].input_mems[1]);
      d2d(in2_mem, net_blocks_cache[0]->stages[0].input_mems[2]);
    }

    net_launch(net_blocks_cache[idx]);
    out_mem = out0_mem;
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], token_offset, out1_mem, 0,
                       bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], token_offset, out2_mem, 0,
                       bytes);
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
  pybind11::class_<Qwen2_5O>(m, "Qwen2_5O")
      .def(pybind11::init<>())
      .def("init", &Qwen2_5O::init)
      .def("forward_embed", &Qwen2_5O::forward_embed)
      .def("forward_vit", &Qwen2_5O::forward_vit)
      .def("forward_audio", &Qwen2_5O::forward_audio)
      .def("forward_first", &Qwen2_5O::forward_first)
      .def("forward_next", &Qwen2_5O::forward_next)
      .def("video_sync", &Qwen2_5O::video_sync)
      .def("clear_history", &Qwen2_5O::clear_history)
      .def("deinit", &Qwen2_5O::deinit)
      .def_readonly("SEQLEN", &Qwen2_5O::SEQLEN) // read SEQLEN in pipeline.py
      .def_readonly("MAX_PIXELS", &Qwen2_5O::MAX_PIXELS)
      .def_readonly("MAX_PATCHES", &Qwen2_5O::MAX_PATCHES)
      .def_readonly("MAX_INPUT_LENGTH", &Qwen2_5O::MAX_INPUT_LENGTH)
      .def_readonly("PREFILL_KV_LENGTH", &Qwen2_5O::PREFILL_KV_LENGTH)
      .def_readonly("support_history", &Qwen2_5O::support_history)
      .def_readonly("history_length", &Qwen2_5O::history_length);
}
