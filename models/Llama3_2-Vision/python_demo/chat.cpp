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
#include "utils.h"
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

static const float ATTENTION_MASK = -9984.;

class Llama3_2 {
public:
  void init(const std::vector<int> &devid, std::string model_path);
  void deinit();
  int forward_first(std::vector<int> &tokens, std::vector<float> &pixel_values,
                    std::vector<int> &aspect_ratio_ids,
                    std::vector<int> &aspect_ratio_mask,
                    std::vector<int> &cross_attn_mask);
  int forward_next();

  std::mt19937 sgen;
  Llama3_2() : sgen(std::random_device()()) {};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);

  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem);

public:
  int token_length;
  int SEQLEN;      // read from bmodel
  int HIDDEN_SIZE; // read from bmodel
  int NUM_LAYERS;  // read from bmodel
  int NUM_TILES;   // read from bmodel
  int NUM_PATCHES; // read from bmodel
  uint16_t mask_value;
  std::vector<int> visited_tokens;
  std::vector<int> cross_attn_layers = {3, 8, 13, 18, 23, 28, 33, 38};

  // generation
  float temperature;
  float top_p;
  float repeat_penalty;
  int repeat_last_n;
  int max_new_tokens;
  std::string generation_mode;
  std::string prompt_mode;

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_vit;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm, *net_greedy_head, *net_penalty_sample_head;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};

void Llama3_2::net_launch(const bm_net_info_t *net, int stage_idx) {
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

void Llama3_2::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void Llama3_2::init(const std::vector<int> &devices, std::string model_path) {

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
  net_vit = bmrt_get_network_info(p_bmrt, "vit");
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
  net_penalty_sample_head =
      bmrt_get_network_info(p_bmrt, "penalty_sample_head");

  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1];   // real seqlen
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1]; // read hidden size
  NUM_TILES = net_vit->stages[0].output_shapes[0].dims[0];
  NUM_PATCHES = net_vit->stages[0].output_shapes[0].dims[1];
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = (num_nets - 6) / 2;
  if (net_embed->output_dtypes[0] == BM_FLOAT16) {
    mask_value = fp32_to_fp16_bits(ATTENTION_MASK);
  } else if (net_embed->output_dtypes[0] == BM_BFLOAT16) {
    mask_value = fp32_to_bf16_bits(ATTENTION_MASK);
  } else {
    std::cerr << "\nError: Invalid attention dtype\n";
    std::cerr << "Supported dtype are 'BM_FLOAT16' or 'BM_BFLOAT16'\n";
    throw std::runtime_error("Invalid attention dtype");
  }
  // resize
  visited_tokens.resize(SEQLEN);

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
    if (std::find(cross_attn_layers.begin(), cross_attn_layers.end(), i) !=
        cross_attn_layers.end()) {
      assert(addr_mode == net_blocks_cache[i]->addr_mode);
      past_key[i] = net_blocks_cache[i]->stages[0].input_mems[2];
      past_value[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    } else {
      assert(addr_mode == net_blocks_cache[i]->addr_mode);
      past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
      past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    }
  }
}

void Llama3_2::deinit() {
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

void Llama3_2::head_launch(const bm_net_info_t *net,
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

int Llama3_2::greedy_search(const bm_net_info_t *net,
                            bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];
  head_launch(net, logits_mem);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

int Llama3_2::penalty_sample(const bm_net_info_t *net,
                             bm_device_mem_t &logits_mem) {
  auto &in1_mem = net->stages[0].input_mems[1];
  auto &in2_mem = net->stages[0].input_mems[2];
  auto &in3_mem = net->stages[0].input_mems[3];
  auto &in4_mem = net->stages[0].input_mems[4];
  auto &out0_mem = net->stages[0].output_mems[0];
  auto &out1_mem = net->stages[0].output_mems[1];

  // repeat_penalty + top_p + top_k + temperature
  std::vector<int> generated_tokens(SEQLEN, visited_tokens[token_length - 1]);
  repeat_last_n = std::min(repeat_last_n, token_length);
  std::copy(visited_tokens.begin() + token_length - repeat_last_n,
            visited_tokens.begin() + token_length, generated_tokens.begin());
  bm_memcpy_s2d(bm_handle, in1_mem, (void *)generated_tokens.data());
  bm_memcpy_s2d(bm_handle, in2_mem, (void *)&top_p);
  bm_memcpy_s2d(bm_handle, in3_mem, (void *)&temperature);
  bm_memcpy_s2d(bm_handle, in4_mem, (void *)&repeat_penalty);

  // inference
  head_launch(net, logits_mem);

  // get logit & token
  int candidate_num = net->stages[0].output_shapes[0].dims[1];
  std::vector<float> probs(candidate_num);
  bm_memcpy_d2s(bm_handle, probs.data(), out0_mem);
  std::vector<int> tokens(candidate_num);
  bm_memcpy_d2s(bm_handle, tokens.data(), out1_mem);

  // penalty_sample
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return tokens[dist(sgen)];
}

int Llama3_2::forward_first(std::vector<int> &tokens,
                            std::vector<float> &pixel_values,
                            std::vector<int> &aspect_ratio_ids,
                            std::vector<int> &aspect_ratio_mask,
                            std::vector<int> &cross_attn_mask) {
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> text_row_mask(SEQLEN, 0);
  std::vector<uint16_t> cross_attention_mask(SEQLEN * NUM_TILES * NUM_PATCHES,
                                             mask_value);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, mask_value);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());
  token_length = tokens.size();

  // valid text token start from 6
  for (int i = 6; i < token_length; i++) {
    text_row_mask[i] = net_embed->output_dtypes[0] == BM_FLOAT16
                           ? fp32_to_bf16_bits(1.)
                           : fp32_to_fp16_bits(1.);
  }
  for (int i = 0; i < SEQLEN; i++) {
    for (int j = 0; j < NUM_TILES; j++) {
      for (int k = 0; k < NUM_PATCHES; k++) {
        if (i < 6)
          cross_attention_mask[i * NUM_TILES * NUM_PATCHES + j * NUM_PATCHES +
                               k] = 0;
        else if (i >= 6 && i < token_length)
          cross_attention_mask[i * NUM_TILES * NUM_PATCHES + j * NUM_PATCHES +
                               k] =
              cross_attn_mask[i * NUM_TILES + j] == 1 ? 0 : mask_value;
        else
          cross_attention_mask[i * NUM_TILES * NUM_PATCHES + j * NUM_PATCHES +
                               k] = mask_value;
      }
    }
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

  // forward vision transformer
  auto &vit_in0_mem = net_vit->stages[0].input_mems[0];
  auto &vit_in1_mem = net_vit->stages[0].input_mems[1];
  auto &vit_in2_mem = net_vit->stages[0].input_mems[2];
  auto &vit_out_mem = net_vit->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, vit_in0_mem, (void *)pixel_values.data());
  bm_memcpy_s2d(bm_handle, vit_in1_mem, (void *)aspect_ratio_ids.data());
  bm_memcpy_s2d(bm_handle, vit_in2_mem, (void *)aspect_ratio_mask.data());
  net_launch(net_vit);

  // forward embeding
  auto in_mem = net_embed->stages[0].input_mems[0];
  auto out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)visited_tokens.data());
  net_launch(net_embed);

  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    if (std::find(cross_attn_layers.begin(), cross_attn_layers.end(), idx) !=
        cross_attn_layers.end()) {
      auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
      auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
      auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
      auto &in3_mem = net_blocks[idx]->stages[0].input_mems[3];
      d2d(in0_mem, out_mem);
      if (idx == cross_attn_layers[0]) {
        d2d(in1_mem, vit_out_mem);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)text_row_mask.data());
        bm_memcpy_s2d(bm_handle, in3_mem, (void *)cross_attention_mask.data());
      }
    } else {
      auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
      auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
      auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
      d2d(in0_mem, out_mem);
      if (idx == 0) {
        // only first time need copy
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      }
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
  if (generation_mode == "greedy") {
    token = greedy_search(net_greedy_head, lm_out_mem);
  } else if (generation_mode == "penalty_sample") {
    token = penalty_sample(net_penalty_sample_head, lm_out_mem);
  }

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

int Llama3_2::forward_next() {
  int cur_token = visited_tokens[token_length - 1];

  std::vector<uint16_t> cross_attention_mask(NUM_TILES * NUM_PATCHES,
                                             mask_value);
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = 0; i < NUM_TILES; i++) {
    for (int j = 0; j < NUM_PATCHES; j++) {
      cross_attention_mask[i * NUM_PATCHES + j] = i < 2 ? 0 : mask_value;
    }
  }
  for (int i = token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = mask_value;
  }
  int32_t position_id = token_length - 1;

  // embedding
  auto in_mem = net_embed_cache->stages[0].input_mems[0];
  auto out_mem = net_embed_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)&cur_token);
  net_launch(net_embed_cache);

  // blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (token_length - 1) * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    if (std::find(cross_attn_layers.begin(), cross_attn_layers.end(), idx) !=
        cross_attn_layers.end()) {
      auto &in0_mem = net_blocks_cache[idx]->stages[0].input_mems[0];
      auto &in1_mem = net_blocks_cache[idx]->stages[0].input_mems[1];
      auto &out0_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
      d2d(in0_mem, out_mem);
      if (idx == cross_attn_layers[0]) {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)cross_attention_mask.data());
      } else {
        d2d(in1_mem,
            net_blocks_cache[cross_attn_layers[0]]->stages[0].input_mems[1]);
      }
      net_launch(net_blocks_cache[idx]);
      out_mem = out0_mem;
    } else {
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
  }

  // forward lmhead
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm);

  int token = 0;
  if (generation_mode == "greedy") {
    token = greedy_search(net_greedy_head, lm_out_mem);
  } else if (generation_mode == "penalty_sample") {
    token = penalty_sample(net_penalty_sample_head, lm_out_mem);
  }

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Llama3_2>(m, "Llama3_2")
      .def(pybind11::init<>())
      .def("init", &Llama3_2::init)
      .def("forward_first", &Llama3_2::forward_first)
      .def("forward_next", &Llama3_2::forward_next)
      .def("deinit", &Llama3_2::deinit)
      .def_readwrite("SEQLEN", &Llama3_2::SEQLEN) // read SEQLEN in pipeline.py
      .def_readwrite("token_length", &Llama3_2::token_length)
      .def_readwrite("temperature", &Llama3_2::temperature)
      .def_readwrite("top_p", &Llama3_2::top_p)
      .def_readwrite("repeat_penalty", &Llama3_2::repeat_penalty)
      .def_readwrite("repeat_last_n", &Llama3_2::repeat_last_n)
      .def_readwrite("max_new_tokens", &Llama3_2::max_new_tokens)
      .def_readwrite("generation_mode", &Llama3_2::generation_mode)
      .def_readwrite("prompt_mode", &Llama3_2::prompt_mode);
}
