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

class Gemma3 {
public:
  void init(const std::vector<int> &devid, std::string model_path);
  void deinit();
  void forward_embed(std::vector<int> &tokens);
  void forward_vit(
      py::array_t<float, py::array::c_style | py::array::forcecast> const
          &pixel_values,
      int vit_offset);
  int forward_first();
  int forward_next();
  std::vector<int> generate(std::vector<int> &history_tokens, int EOS);

  std::mt19937 sgen;
  Gemma3() : sgen(std::random_device()()) {};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  void net_launch_dyn(const bm_net_info_t *net, int real_len,
                      int stage_idx = 0);
  void net_launch_decode(int block_idx, int kv_offset,
                         bm_device_mem_t &input_mem, const int *pos_id,
                         std::vector<uint16_t> &attention_mask);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset = 0,
                  int size = 0);
  void init_by_names();
  int greedy_search(bm_device_mem_t &logits_mem);
  int penalty_sample(bm_device_mem_t &logits_mem);

public:
  int hidden_bytes;
  int kv_bytes;
  int token_length;
  int SEQLEN; // read from bmodel
  int HIDDEN_SIZE;
  int NUM_LAYERS; // read from bmodel
  bool lmhead_with_topk;
  bool is_dynamic;
  std::vector<int> visited_tokens;
  uint16_t mask_value;

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
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_vit;
  bm_device_mem_t dev_buffer;
  const bm_net_info_t *net_lm, *net_greedy_head, *net_sample_head;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};

void Gemma3::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
                 int size) {
  if (!size)
    size = bm_mem_get_device_size(src);
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void Gemma3::init_by_names() {
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
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  const char **net_names = nullptr;
  auto num_nets = bmrt_get_network_number(p_bmrt);
  bmrt_get_network_names(p_bmrt, &net_names);
  net_greedy_head = nullptr;
  auto num_blocks = num_nets - 3; // 3 nets are embed, lm_head, embedding_cache
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
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
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
  if (net_blocks[0]->input_dtypes[0] == BM_FLOAT16) {
    mask_value = 0xF0E2;
  } else if (net_blocks[0]->input_dtypes[0] == BM_BFLOAT16) {
    mask_value = 0xC61C;
  } else {
    std::cerr << "\nError: Invalid attention dtype\n";
    std::cerr << "Supported dtype are 'BM_FLOAT16' or 'BM_BFLOAT16'\n";
    throw std::runtime_error("Invalid attention dtype");
  }
}

void Gemma3::forward_embed(std::vector<int> &tokens) {
  std::vector<int> input_ids(SEQLEN, 0);
  std::copy(tokens.begin(), tokens.end(), input_ids.data());

  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)input_ids.data());
  net_launch(net_embed);
  d2d(dev_buffer, out_mem);
  token_length = tokens.size();
}

void Gemma3::forward_vit(
    py::array_t<float, py::array::c_style | py::array::forcecast> const
        &pixel_values,
    int vit_offset) {
  auto &vit_in_mem = net_vit->stages[0].input_mems[0];
  auto &vit_out_mem = net_vit->stages[0].output_mems[0];
  auto h = net_vit->stages[0].input_shapes[0].dims[2];
  auto w = net_vit->stages[0].input_shapes[0].dims[3];
  assert(pixel_values.size() == 1 * 3 * h * w);
  auto buf = pixel_values.request();
  float *ptr = static_cast<float *>(buf.ptr);
  bm_memcpy_s2d(bm_handle, vit_in_mem, (void *)ptr);
  // launch vit
  net_launch(net_vit);

  // concatenante texting embedding and image embedding
  int dst_offset = vit_offset * HIDDEN_SIZE * sizeof(uint16_t);
  auto vit_size = bm_mem_get_device_size(vit_out_mem);
  bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset, vit_out_mem, 0,
                     vit_size);
}

void Gemma3::init(const std::vector<int> &devices, std::string model_path) {

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
  bool ret = false;
  ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");

  init_by_names();

  // resize
  visited_tokens.resize(SEQLEN);

  hidden_bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[0]);
  kv_bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  is_dynamic = net_blocks[0]->is_dynamic;
  auto addr_mode = net_blocks_cache[0]->addr_mode;
  for (int i = 0; i < NUM_LAYERS; i++) {
    assert(addr_mode == net_blocks_cache[i]->addr_mode);
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
  }
  auto buffer_size =
      bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  auto status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  assert(BM_SUCCESS == status);
}

void Gemma3::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

void Gemma3::net_launch(const bm_net_info_t *net, int stage_idx) {
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

void Gemma3::net_launch_dyn(const bm_net_info_t *net, int real_len,
                            int stage_idx) {
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

  in_tensors[0].shape.dims[1] = real_len;
  in_tensors[1].shape.dims[1] = real_len;
  in_tensors[2].shape.dims[2] = real_len;
  in_tensors[2].shape.dims[3] = real_len;

  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  // bm_thread_sync(bm_handle);
}

void Gemma3::net_launch_decode(int idx, int kv_offset,
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
      past_key[idx].u.device.device_addr + kv_offset, kv_bytes);
  auto v_mem = bm_mem_from_device(
      past_value[idx].u.device.device_addr + kv_offset, kv_bytes);
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

int Gemma3::greedy_search(bm_device_mem_t &logits_mem) {
  auto &out_mem = net_greedy_head->stages[0].output_mems[0];
  bm_set_device_mem(&net_greedy_head->stages[0].input_mems[0], logits_mem.size,
                    logits_mem.u.device.device_addr);
  net_launch(net_greedy_head);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

int Gemma3::penalty_sample(bm_device_mem_t &logits_mem) {
  auto &in0_mem = net_sample_head->stages[0].input_mems[0];
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
  d2d(in0_mem, logits_mem, 0, bm_mem_get_device_size(logits_mem));
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

int Gemma3::forward_first() {
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, mask_value);

  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  if (is_dynamic) {
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * token_length + j] = 0;
      }
    }
  } else {
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * SEQLEN + j] = 0;
      }
    }
  }
  // empty
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty_net(bm_handle, net_blocks[i]);
    empty_net(bm_handle, net_blocks_cache[i]);
  }

  // forward embeding
  auto out_mem = dev_buffer;
  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    empty(bm_handle, net_blocks[idx]->stages[0].input_mems[0]);
    d2d(in0_mem, out_mem, 0, token_length * hidden_bytes);
    if (idx == 0) {
      // only first time need copy
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    }
    if (is_dynamic) {
      net_launch_dyn(net_blocks[idx], token_length);
    } else {
      net_launch(net_blocks[idx]);
    }
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1], 0,
        token_length * kv_bytes);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2], 0,
        token_length * kv_bytes);
  }

  // forward lmhead
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (token_length - 1) * hidden_bytes, hidden_bytes);
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

int Gemma3::forward_next() {
  int cur_token = visited_tokens[token_length - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
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
  int token_offset = (token_length - 1) * kv_bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    net_launch_decode(idx, token_offset, out_mem, &position_id, attention_mask);
    out_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
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
  pybind11::class_<Gemma3>(m, "Gemma3")
      .def(pybind11::init<>())
      .def("init", &Gemma3::init)
      .def("forward_embed", &Gemma3::forward_embed)
      .def("forward_vit", &Gemma3::forward_vit)
      .def("forward_first", &Gemma3::forward_first)
      .def("forward_next", &Gemma3::forward_next)
      .def("deinit", &Gemma3::deinit)
      .def_readwrite("SEQLEN", &Gemma3::SEQLEN) // read SEQLEN in pipeline.py
      .def_readwrite("token_length", &Gemma3::token_length)
      .def_readwrite("generation_mode", &Gemma3::generation_mode)
      .def_readwrite("penalty", &Gemma3::penalty)
      .def_readwrite("temperature", &Gemma3::temperature)
      .def_readwrite("top_k", &Gemma3::top_k)
      .def_readwrite("top_p", &Gemma3::top_p);
}
