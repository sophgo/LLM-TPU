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

#include <pybind11/numpy.h>

static const int MEDIA_TOKEN_ID = 151667;

class InternVL3 {
public:
  void init(const std::vector<int> &devid, std::string model_path);
  void deinit();
  int forward_first(pybind11::array_t<int> tokens,
                    pybind11::array_t<float> pixel_values);
  int forward_next();

  std::mt19937 sgen;
  InternVL3() : sgen(std::random_device()()) {};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  void init_by_names();
  int greedy_search(bm_device_mem_t &logits_mem);
  int penalty_sample(bm_device_mem_t &logits_mem);

public:
  int token_length;
  int SEQLEN;
  int HIDDEN_SIZE;
  int NUM_LAYERS;
  int NUM_IMAGE_TOKEN;
  uint16_t mask_value;
  bool lmhead_with_topk;
  std::vector<int> visited_tokens;

  // generation
  std::string generation_mode;
  float penalty;
  float temperature;
  int top_k;
  float top_p;

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_vit;
  const bm_net_info_t *net_embed, *net_embed_cache;
  const bm_net_info_t *net_lm, *net_greedy_head, *net_sample_head;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};

void InternVL3::net_launch(const bm_net_info_t *net, int stage_idx) {
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

void InternVL3::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void InternVL3::init(const std::vector<int> &devices, std::string model_path) {

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

  init_by_names();

  if (net_embed_cache->output_dtypes[0] == BM_FLOAT16) {
    mask_value = 0xF0E2; // float16
  } else if (net_embed_cache->output_dtypes[0] == BM_BFLOAT16) {
    mask_value = 0xC61C; // -9984 by bfloat16
  } else {
    std::cerr << "\nError: Invalid attention dtype\n";
    std::cerr << "Supported dtype are 'BM_FLOAT16' or 'BM_BFLOAT16'\n";
    throw std::runtime_error("Invalid attention dtype");
  }

  // resize
  visited_tokens.resize(SEQLEN);

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  auto addr_mode = net_blocks_cache[0]->addr_mode;
  for (int i = 0; i < NUM_LAYERS; i++) {
    assert(addr_mode == net_blocks_cache[i]->addr_mode);
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
  }

  auto buffer_size =
      bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
}

void InternVL3::init_by_names() {
  auto is_exist = [](const char *name, const char **names, int num) {
    for (int i = 0; i < num; i++) {
      if (strcmp(name, names[i]) == 0) {
        return true;
      }
    }
    return false;
  };
  net_vit = bmrt_get_network_info(p_bmrt, "vit");
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  const char **net_names = nullptr;
  auto num_nets = bmrt_get_network_number(p_bmrt);
  bmrt_get_network_names(p_bmrt, &net_names);
  net_greedy_head = nullptr;
  auto num_blocks = num_nets - 4; // 3 nets are embed, lm_head, embedding_cache
  if (is_exist("greedy_head", net_names, num_nets)) {
    net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
    num_blocks--; // greedy_head is not a block
  }
  net_sample_head = nullptr;
  if (is_exist("sample_head", net_names, num_nets)) {
    net_sample_head = bmrt_get_network_info(p_bmrt, "sample_head");
    num_blocks--; // sample_head is not a block
  }

  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1];
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  NUM_IMAGE_TOKEN = net_vit->stages[0].output_shapes[0].dims[0];
  lmhead_with_topk = net_lm->stages[0].output_shapes[0].dims[1] == 1;

  NUM_LAYERS = num_blocks / 2;

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
}

void InternVL3::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

int InternVL3::greedy_search(bm_device_mem_t &logits_mem) {
  auto &out_mem = net_greedy_head->stages[0].output_mems[0];
  bm_set_device_mem(&net_greedy_head->stages[0].input_mems[0], logits_mem.size,
                    logits_mem.u.device.device_addr);
  net_launch(net_greedy_head);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

int InternVL3::penalty_sample(bm_device_mem_t &logits_mem) {
  auto &in1_mem = net_sample_head->stages[0].input_mems[1];
  auto &in2_mem = net_sample_head->stages[0].input_mems[2];
  auto &in3_mem = net_sample_head->stages[0].input_mems[3];
  auto &in4_mem = net_sample_head->stages[0].input_mems[4];
  auto &in5_mem = net_sample_head->stages[0].input_mems[5];
  auto &out0_mem = net_sample_head->stages[0].output_mems[0];
  auto &out1_mem = net_sample_head->stages[0].output_mems[1];

  // repeat_penalty + top_p + top_k + temperature
  bm_memcpy_s2d(bm_handle, in1_mem, (void *)visited_tokens.data());
  bm_memcpy_s2d(bm_handle, in2_mem, (void *)&penalty);
  bm_memcpy_s2d(bm_handle, in3_mem, (void *)&temperature);
  bm_memcpy_s2d(bm_handle, in4_mem, (void *)&top_k);
  bm_memcpy_s2d(bm_handle, in5_mem, (void *)&top_p);

  // inference
  bm_set_device_mem(&net_sample_head->stages[0].input_mems[0], logits_mem.size,
                    logits_mem.u.device.device_addr);
  net_launch(net_sample_head);

  // get logit & token
  int candidate_num = top_k;
  std::vector<float> probs(candidate_num);
  bm_memcpy_d2s_partial_offset(bm_handle, probs.data(), out0_mem,
                               top_k * sizeof(float), 0);
  std::vector<int> tokens(candidate_num);
  bm_memcpy_d2s_partial_offset(bm_handle, tokens.data(), out1_mem,
                               top_k * sizeof(float), 0);

  // sample
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return tokens[dist(sgen)];
}

int InternVL3::forward_first(pybind11::array_t<int> tokens,
                             pybind11::array_t<float> pixel_values) {
  auto tokens_buf = tokens.request();
  int *tokens_ptr = static_cast<int *>(tokens_buf.ptr);
  size_t tokens_len = tokens_buf.size;

  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, mask_value);

  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  std::copy(tokens_ptr, tokens_ptr + tokens_len, visited_tokens.data());
  token_length = tokens_len;

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
  d2d(dev_buffer, out_mem);

  int bytes = out_mem.size / SEQLEN;
  if (pixel_values.size() > 0) {
    auto pixel_buf = pixel_values.request();
    float *pixel_ptr = static_cast<float *>(pixel_buf.ptr);
    size_t pixel_len = pixel_buf.size;

    int vit_offset = 0;
    for (int i = 0; i < token_length; i++) {
      if (visited_tokens[i] == MEDIA_TOKEN_ID) {
        vit_offset = i;
        break;
      }
    }
    auto &vit_in_mem = net_vit->stages[0].input_mems[0];
    auto &vit_out_mem = net_vit->stages[0].output_mems[0];
    int pixel_bytes = vit_in_mem.size / sizeof(float);
    int num_patches = pixel_len / pixel_bytes;
    for (int i = 0; i < num_patches; i++) {
      bm_memcpy_s2d(bm_handle, vit_in_mem,
                    (void *)(pixel_ptr + i * pixel_bytes));
      net_launch(net_vit);
      bm_memcpy_d2d_byte(bm_handle, dev_buffer,
                         (vit_offset + i * NUM_IMAGE_TOKEN) * bytes,
                         vit_out_mem, 0, NUM_IMAGE_TOKEN * bytes);
    }
  }

  // forward blocks
  out_mem = dev_buffer;
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

  if (lmhead_with_topk) {
    bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  } else if (generation_mode == "greedy") {
    token = greedy_search(lm_out_mem);
  } else if (generation_mode == "sample") {
    token = penalty_sample(lm_out_mem);
  }

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

int InternVL3::forward_next() {
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
    auto &in3_mem = net_blocks_cache[idx]->stages[0].input_mems[3];
    auto &in4_mem = net_blocks_cache[idx]->stages[0].input_mems[4];
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
  if (lmhead_with_topk) {
    bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  } else if (generation_mode == "greedy") {
    token = greedy_search(lm_out_mem);
  } else if (generation_mode == "sample") {
    token = penalty_sample(lm_out_mem);
  }

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<InternVL3>(m, "InternVL3")
      .def(pybind11::init<>())
      .def("init", &InternVL3::init)
      .def("forward_first", &InternVL3::forward_first)
      .def("forward_next", &InternVL3::forward_next)
      .def("deinit", &InternVL3::deinit)
      .def_readwrite("SEQLEN", &InternVL3::SEQLEN)
      .def_readwrite("NUM_IMAGE_TOKEN", &InternVL3::NUM_IMAGE_TOKEN)
      .def_readwrite("token_length", &InternVL3::token_length)
      .def_readwrite("generation_mode", &InternVL3::generation_mode)
      .def_readwrite("penalty", &InternVL3::penalty)
      .def_readwrite("temperature", &InternVL3::temperature)
      .def_readwrite("top_k", &InternVL3::top_k)
      .def_readwrite("top_p", &InternVL3::top_p);
}
