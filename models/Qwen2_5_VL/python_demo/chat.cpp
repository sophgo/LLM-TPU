//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdio.h>
#include <vector>

#include "utils.h"

static const uint16_t ATTENTION_MASK = 0xC61C; // -9984 by bfloat16

class Qwen2VL {
public:
  void init(int devid, std::string model_path);
  void deinit();
  int forward_first(std::vector<int> &tokens, std::vector<float> &pixel_values,
                    std::vector<int> &grid_thw, int vit_offset,
                    int valid_vit_length);
  int forward_next();

  std::mt19937 sgen;
  Qwen2VL() : sgen(std::random_device()()) {};

private:
  void make_vit_posid(std::vector<int> &grid_thw,
                      std::vector<int> &position_ids);
  void make_vit_attn_mask(std::vector<int> &grid_thw,
                          std::vector<float> &attention_mask);
  std::vector<int> make_posid(const std::vector<int> &grid_thw, int vit_offset,
                              int valid_vit_length, int token_length);

  void vit_launch(std::vector<float> &pixel_values, int vit_offset,
                  int valid_vit_length, std::vector<int> &grid_thw,
                  bm_device_mem_t &out_mem);
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  uint16_t mask_value;

public:
  int token_length;
  int SEQLEN; // read from bmodel
  int HIDDEN_SIZE;
  int NUM_LAYERS; // read from bmodel
  uint64_t VIT_DIMS;
  std::string generation_mode;
  int MAX_PIXELS;
  int max_pos;
  int spatial_merge_size;

private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm;
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
  bm_thread_sync(bm_handle);
}

void Qwen2VL::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void Qwen2VL::init(int dev_id, std::string model_path) {

  // request bm_handle
  std::cout << "Device [ " << dev_id << " ] loading .....\n";
  bm_status_t status = bm_dev_request(&bm_handle, dev_id);
  assert(BM_SUCCESS == status);

  // create bmruntime
  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);

  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");

  // net embed and lm_head
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_vit = bmrt_get_network_info(p_bmrt, "vit");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1]; // real seqlen
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  MAX_PIXELS = net_vit->stages[0].input_shapes[0].dims[0];
  VIT_DIMS = net_vit->stages[0].input_shapes[0].dims[1];
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = (num_nets - 4) / 2;
  printf("Num Layers:%d\n", NUM_LAYERS);
  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }

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

void Qwen2VL::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
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
  bm_thread_sync(bm_handle);
}

void Qwen2VL::make_vit_posid(std::vector<int> &grid_thw,
                             std::vector<int> &pos_ids) {
  int t = grid_thw[0];
  int h = grid_thw[1];
  int w = grid_thw[2];

  // generate hpos_ids
  std::vector<int> hpos_ids;
  for (int n = 0; n < h; n += spatial_merge_size) {
    for (int _ = 0; _ < w / spatial_merge_size; ++_) {
      hpos_ids.push_back(n);
      hpos_ids.push_back(n);
      hpos_ids.push_back(n + 1);
      hpos_ids.push_back(n + 1);
    }
  }

  // generate wpos_ids
  std::vector<int> wpos_ids;
  for (int _ = 0; _ < h / spatial_merge_size; ++_) {
    for (int e = 0; e < w; e += spatial_merge_size) {
      wpos_ids.push_back(e);
      wpos_ids.push_back(e + 1);
      wpos_ids.push_back(e);
      wpos_ids.push_back(e + 1);
    }
  }

  int valid_vit_pixels = h * w;
  for (int i = 0; i < t; ++i) {
    for (int j = 0; j < valid_vit_pixels; ++j) {
      pos_ids[i * valid_vit_pixels + 2 * j] = hpos_ids[j];
      pos_ids[i * valid_vit_pixels + 2 * j + 1] = wpos_ids[j];
    }
  }
}

void Qwen2VL::make_vit_attn_mask(std::vector<int> &grid_thw,
                                 std::vector<float> &attention_mask) {
  // Extract t, h, w from grid_thw
  int t = grid_thw[0];
  int h = grid_thw[1];
  int w = grid_thw[2];

  // Compute cu_seqlens
  std::vector<int> cu_seqlens(t + 1, 0);
  for (int i = 0; i <= t; ++i) {
    cu_seqlens[i] = h * w * i;
  }

  // Update attention_mask based on cu_seqlens
  for (size_t i = 1; i < cu_seqlens.size(); ++i) {
    int start = cu_seqlens[i - 1];
    int end = cu_seqlens[i];
    for (int row = start; row < end; ++row) {
      for (int col = start; col < end; ++col) {
        size_t index = row * MAX_PIXELS + col;
        if (index < attention_mask.size()) {
          attention_mask[index] = 0;
        }
      }
    }
  }
}

void Qwen2VL::vit_launch(std::vector<float> &pixel_values, int vit_offset,
                         int valid_vit_length, std::vector<int> &grid_thw,
                         bm_device_mem_t &out_mem) {
  d2d(dev_buffer, out_mem);
  out_mem = dev_buffer;
  // forward vision transformer
  std::vector<float> pixel_values_pad(MAX_PIXELS * VIT_DIMS, 0);
  // Initialize attention_mask with -10000
  std::vector<float> attention_mask(MAX_PIXELS * MAX_PIXELS, -10000.0f);
  std::vector<int> position_ids(MAX_PIXELS * 2, 0);
  make_vit_posid(grid_thw, position_ids);
  make_vit_attn_mask(grid_thw, attention_mask);
  std::copy(pixel_values.begin(), pixel_values.end(), pixel_values_pad.data());

  empty_net(bm_handle, net_vit);

  auto &vit_in0_mem = net_vit->stages[0].input_mems[0];
  auto &vit_in1_mem = net_vit->stages[0].input_mems[1];
  auto &vit_in2_mem = net_vit->stages[0].input_mems[2];
  auto &vit_out_mem = net_vit->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, vit_in0_mem, (void *)pixel_values_pad.data());
  bm_memcpy_s2d(bm_handle, vit_in1_mem, (void *)position_ids.data());
  bm_memcpy_s2d(bm_handle, vit_in2_mem, (void *)attention_mask.data());
  net_launch(net_vit);

  // concatenante texting embedding and image embedding
  int dst_offset = vit_offset * HIDDEN_SIZE * sizeof(uint16_t);
  int vit_size = valid_vit_length * HIDDEN_SIZE * sizeof(uint16_t);
  bm_memcpy_d2d_byte(bm_handle, out_mem, dst_offset, vit_out_mem, 0, vit_size);
}

std::vector<int> Qwen2VL::make_posid(const std::vector<int> &grid_thw,
                                     int vit_offset, int valid_vit_length,
                                     int token_length) {
  int text_len = vit_offset;

  // Assuming grid_thw has at least one element
  int llm_grid_t = grid_thw[0];
  int llm_grid_h = grid_thw[1] / spatial_merge_size;
  int llm_grid_w = grid_thw[2] / spatial_merge_size;

  std::vector<int> t_position_ids;
  std::vector<int> h_position_ids;
  std::vector<int> w_position_ids;

  // Populate t_position_ids
  for (int i = text_len; i < llm_grid_t + text_len; ++i) {
    for (int j = 0; j < llm_grid_h * llm_grid_w; ++j) {
      t_position_ids.push_back(i);
    }
  }

  // Populate h_position_ids
  for (int _ = 0; _ < llm_grid_t; ++_) {
    for (int i = 0; i < llm_grid_h; ++i) {
      for (int j = 0; j < llm_grid_w; ++j) {
        h_position_ids.push_back(i + text_len);
      }
    }
  }

  // Populate w_position_ids
  for (int _ = 0; _ < llm_grid_t; ++_) {
    for (int i = 0; i < llm_grid_h; ++i) {
      for (int j = text_len; j < llm_grid_w + text_len; ++j) {
        w_position_ids.push_back(j);
      }
    }
  }

  // Calculate starting index for tail text length
  int st_idx = w_position_ids.back() + 1;
  int tail_text_len = token_length - valid_vit_length - text_len;

  // Prepare final position ids
  std::vector<int> position_ids;
  position_ids.reserve(SEQLEN * 3);

  // Prepare head position ids
  std::vector<int> head_position_ids;
  for (int i = 0; i < text_len; ++i) {
    head_position_ids.push_back(i);
  }

  // Prepare tail position ids
  std::vector<int> tail_position_ids;
  for (int i = st_idx; i < st_idx + tail_text_len; ++i) {
    tail_position_ids.push_back(i);
  }

  // Fill position_ids for t
  position_ids.insert(
      position_ids.end(), head_position_ids.begin(),
      head_position_ids.end()); // Fill with 0 for range text_len
  position_ids.insert(position_ids.end(), t_position_ids.begin(),
                      t_position_ids.end());
  position_ids.insert(position_ids.end(), tail_position_ids.begin(),
                      tail_position_ids.end());
  position_ids.insert(position_ids.end(), SEQLEN - token_length,
                      1); // Fill with 1

  // Fill position_ids for h
  position_ids.insert(
      position_ids.end(), head_position_ids.begin(),
      head_position_ids.end()); // Fill with 0 for range text_len
  position_ids.insert(position_ids.end(), h_position_ids.begin(),
                      h_position_ids.end());
  position_ids.insert(position_ids.end(), tail_position_ids.begin(),
                      tail_position_ids.end());
  position_ids.insert(position_ids.end(), SEQLEN - token_length,
                      1); // Fill with 1

  // Fill position_ids for w
  position_ids.insert(
      position_ids.end(), head_position_ids.begin(),
      head_position_ids.end()); // Fill with 0 for range text_len
  position_ids.insert(position_ids.end(), w_position_ids.begin(),
                      w_position_ids.end());
  position_ids.insert(position_ids.end(), tail_position_ids.begin(),
                      tail_position_ids.end());
  position_ids.insert(position_ids.end(), SEQLEN - token_length,
                      1); // Fill with 1

  max_pos = st_idx + tail_text_len - 1;

  return position_ids;
}

int Qwen2VL::forward_first(std::vector<int> &tokens,
                           std::vector<float> &pixel_values,
                           std::vector<int> &grid_thw, int vit_offset,
                           int valid_vit_length) {
  std::vector<int> input_ids(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  std::copy(tokens.begin(), tokens.end(), input_ids.data());

  token_length = tokens.size(); // text input length

  auto position_ids =
      make_posid(grid_thw, vit_offset, valid_vit_length, token_length);
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < token_length; j++) {
      if (j <= i) {
        attention_mask[i * SEQLEN + j] = 0;
      } 
    }
  }

  // forward embeding
  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)input_ids.data());
  net_launch(net_embed); // prefil embedding

  if (pixel_values.size() > 0) {
    vit_launch(pixel_values, vit_offset, valid_vit_length, grid_thw, out_mem);
  }

  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    // d2d(in0_mem, block_out_mem);
    d2d(in0_mem, out_mem);
    if (idx == 0) {
      // only first time need copy
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_ids.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    }
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2]);
  }

  // forward lmhead
  int bytes = out_mem.size / SEQLEN;
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (token_length - 1) * bytes, bytes);
  net_launch(net_lm);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  token_length++;
  return token;
}

int Qwen2VL::forward_next() {
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = ATTENTION_MASK;
  }
  max_pos++;
  std::vector<int> token_pos = {max_pos, max_pos, max_pos};

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
  int token_offset = (token_length - 1) * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks_cache[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks_cache[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks_cache[idx]->stages[0].input_mems[2];
    auto &out0_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
    auto &out1_mem = net_blocks_cache[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks_cache[idx]->stages[0].output_mems[2];
    d2d(in0_mem, out_mem);
    if (idx == 0) {
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)token_pos.data());
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
  bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  token_length++;
  return token;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Qwen2VL>(m, "Qwen2VL")
      .def(pybind11::init<>())
      .def("init", &Qwen2VL::init)
      .def("forward_first", &Qwen2VL::forward_first)
      .def("forward_next", &Qwen2VL::forward_next)
      .def("deinit", &Qwen2VL::deinit)
      .def_readwrite("SEQLEN", &Qwen2VL::SEQLEN) // read SEQLEN in pipeline.py
      .def_readwrite("MAX_PIXELS", &Qwen2VL::MAX_PIXELS)
      .def_readwrite("spatial_merge_size", &Qwen2VL::spatial_merge_size)
      .def_readwrite("token_length", &Qwen2VL::token_length)
      .def_readwrite("generation_mode", &Qwen2VL::generation_mode);
}
