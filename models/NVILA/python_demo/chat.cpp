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

static const int MEDIA_TOKEN_ID = 151649;

class NVILA {
public:
  void init(const std::vector<int> &devid, std::string model_path);
  void deinit();
  std::vector<int16_t> forward_vit(std::vector<float> &pixel_values);
  std::vector<int16_t> forward_projector(std::vector<int16_t> &image_feature);
  void forward_vit_projector(std::vector<float> &pixel_values);
  int forward_first(std::vector<int> &tokens,
                    std::vector<int16_t> &media_embeds);
  int forward_next();

  std::mt19937 sgen;
  NVILA() : sgen(std::random_device()()) {};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);

public:
  int token_length;
  int SEQLEN;
  int HIDDEN_SIZE;
  int NUM_LAYERS;
  uint16_t mask_value;
  bool forward_vit_mm;
  std::vector<int> visited_tokens;

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_vit, *net_projector, *net_vit_mm;
  const bm_net_info_t *net_embed, *net_embed_cache;
  const bm_net_info_t *net_lm;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};

void NVILA::net_launch(const bm_net_info_t *net, int stage_idx) {
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

void NVILA::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void NVILA::init(const std::vector<int> &devices, std::string model_path) {

  // request bm_handle
  std::cout << "Device [ ";
  for (auto d : devices) {
    std::cout << d << " ";
  }
  std::cout << "] loading ....\n";
  for (auto d : devices) {
    bm_handle_t h;
    bm_status_t status = bm_dev_request(&h, d);
    assert(BM_SUCCESS == status);
    handles.push_back(h);
  }
  bm_handle = handles[0];

  // create bmruntime
#ifdef SOC_TARGET
  p_bmrt = bmrt_create(handles[0]);
#else
  p_bmrt = bmrt_create_ex(handles.data(), handles.size());
#endif
  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");

  // net embed and lm_head
  net_vit_mm = bmrt_get_network_info(p_bmrt, "vitmm");
  net_vit = bmrt_get_network_info(p_bmrt, "vit");
  net_projector = bmrt_get_network_info(p_bmrt, "projector");
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");

  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1];   // real seqlen
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1]; // read hidden size
  auto num_nets = bmrt_get_network_number(p_bmrt);
  if (net_vit_mm) {
    NUM_LAYERS = (num_nets - 6) / 2;
    auto buffer_size =
        bm_mem_get_device_size(net_vit_mm->stages[0].output_mems[0]);
    bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  } else
    NUM_LAYERS = (num_nets - 5) / 2;

  if (net_blocks_cache[0]->output_dtypes[0] == BM_FLOAT16) {
    mask_value = 0xF0E2; // float16
  } else if (net_blocks_cache[0]->output_dtypes[0] == BM_BFLOAT16) {
    mask_value = 0xC61C; // -9984 by bfloat16
  } else {
    std::cerr << "\nError: Invalid attention dtype\n";
    std::cerr << "Supported dtype are 'BM_FLOAT16' or 'BM_BFLOAT16'\n";
    throw std::runtime_error("Invalid attention dtype");
  }

  // resize
  visited_tokens.resize(SEQLEN);
  forward_vit_mm = false;

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
  auto addr_mode = net_blocks_cache[0]->addr_mode;
  for (int i = 0; i < NUM_LAYERS; i++) {
    assert(addr_mode == net_blocks_cache[i]->addr_mode);
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
  }
}

void NVILA::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

std::vector<int16_t> NVILA::forward_vit(std::vector<float> &pixel_values) {
  auto &in_mem = net_vit->stages[0].input_mems[0];
  auto &out_mem = net_vit->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)pixel_values.data());
  net_launch(net_vit);
  int cnt = out_mem.size / sizeof(uint16_t);
  std::vector<int16_t> image_feature(cnt);
  bm_memcpy_d2s(bm_handle, image_feature.data(), out_mem);
  return image_feature;
}

std::vector<int16_t>
NVILA::forward_projector(std::vector<int16_t> &image_feature) {
  auto &in_mem = net_projector->stages[0].input_mems[0];
  auto &out_mem = net_projector->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)image_feature.data());
  net_launch(net_projector);
  int cnt = out_mem.size / sizeof(uint16_t);
  std::vector<int16_t> media_embeds(cnt);
  bm_memcpy_d2s(bm_handle, media_embeds.data(), out_mem);
  return media_embeds;
}

void NVILA::forward_vit_projector(std::vector<float> &pixel_values) {
  auto &in_mem = net_vit_mm->stages[0].input_mems[0];
  auto &out_mem = net_vit_mm->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)pixel_values.data());
  net_launch(net_vit_mm);
  d2d(dev_buffer, out_mem);
  forward_vit_mm = true;
}

int NVILA::forward_first(std::vector<int> &tokens,
                         std::vector<int16_t> &media_embeds) {
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, mask_value);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());
  token_length = tokens.size();

  int vit_offset = 0;
  int visual_length = 0;
  if (!media_embeds.empty() || forward_vit_mm) {
    if (forward_vit_mm) {
      auto &vit_out_mem = net_vit_mm->stages[0].output_mems[0];
      visual_length = vit_out_mem.size / HIDDEN_SIZE / 2;
    } else {
      visual_length = media_embeds.size() / HIDDEN_SIZE / 2;
    }
    for (int i = 0; i < token_length; i++) {
      if (visited_tokens[i] == MEDIA_TOKEN_ID) {
        vit_offset = i;
        int src_start = i + 1;
        int src_end = token_length;
        int dst_start = i + visual_length;
        std::copy(visited_tokens.begin() + src_start,
                  visited_tokens.begin() + src_end,
                  visited_tokens.begin() + dst_start);
        break;
      }
    }
    token_length += visual_length - 1;
  }

  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < SEQLEN; j++) {
      if (j <= i) {
        attention_mask[i * SEQLEN + j] = 0;
      }
    }
  }

  // forward embeding
  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)visited_tokens.data());
  net_launch(net_embed);

  int bytes = out_mem.size / SEQLEN;
  if (!media_embeds.empty()) {
    bm_memcpy_s2d_partial_offset(bm_handle, out_mem,
                                 (void *)media_embeds.data(),
                                 media_embeds.size(), vit_offset * bytes);
  } else if (forward_vit_mm) {
    bm_memcpy_d2d_byte(bm_handle, out_mem, vit_offset * bytes, dev_buffer, 0,
                       visual_length * bytes);
  }

  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    d2d(in0_mem, out_mem);
    if (idx == 0) {
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    }
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2]);
  }

  // forward lmhead
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (token_length - 1) * bytes, bytes);
  net_launch(net_lm);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

int NVILA::forward_next() {
  int cur_token = visited_tokens[token_length - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = mask_value;
  }
  int32_t position_id = token_length - 1;

  // embedding
  auto &in_mem = net_embed_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)&cur_token);
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
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
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
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm);

  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<NVILA>(m, "NVILA")
      .def(pybind11::init<>())
      .def("init", &NVILA::init)
      .def("forward_vit", &NVILA::forward_vit)
      .def("forward_projector", &NVILA::forward_projector)
      .def("forward_vit_projector", &NVILA::forward_vit_projector)
      .def("forward_first", &NVILA::forward_first)
      .def("forward_next", &NVILA::forward_next)
      .def("deinit", &NVILA::deinit)
      .def_readwrite("SEQLEN", &NVILA::SEQLEN)
      .def_readwrite("token_length", &NVILA::token_length);
}
