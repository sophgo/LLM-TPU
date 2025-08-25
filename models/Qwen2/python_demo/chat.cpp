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
#include <dlfcn.h>
#include <getopt.h>
#include <inttypes.h>
#include <iostream>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdio.h>
#include <vector>

static const uint16_t ATTENTION_MASK = 0xC61C;
typedef uint8_t *(*decrypt_func)(const uint8_t *, uint64_t, uint64_t *);

class Qwen {
public:
  void init(const std::vector<int> &devid, std::string model_path);
  void deinit();
  void init_decrypt();
  void deinit_decrypt();
  int forward_first(std::vector<int> &tokens);
  int forward_next();
  std::vector<int> generate(std::vector<int> &history_tokens, int EOS);

  std::mt19937 sgen;
  Qwen() : sgen(std::random_device()()) {};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  void net_launch_dyn(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
                  int size);

  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem);

public:
  int hidden_bytes;
  int kv_bytes;
  int token_length;
  int SEQLEN; // read from bmodel
  int MAX_INPUT_LENGTH;
  int NUM_LAYERS; // read from bmodel
  int TOKEN_LEN;
  bool lmhead_with_topk;
  bool is_dynamic;
  std::vector<int> visited_tokens;
  std::string lib_path;

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
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm, *net_greedy_head, *net_penalty_sample_head;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;

  void *decrypt_handle_;      // handle of decrypt lib
  decrypt_func decrypt_func_; // decrypt func from lib
};

void Qwen::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void Qwen::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset) {
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0,
                     bm_mem_get_device_size(src));
}

void Qwen::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
               int size) {
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void Qwen::init_decrypt() {
  // init decrypt
  if (lib_path.empty()) {
    return;
  }
  decrypt_handle_ = dlopen(lib_path.c_str(), RTLD_LAZY);
  if (!decrypt_handle_) {
    std::cout << "Error:" << "Decrypt lib [" << lib_path << "] load failed."
              << std::endl;
    return;
  }
  decrypt_func_ = (decrypt_func)dlsym(decrypt_handle_, "decrypt");
  auto error = dlerror();
  if (error) {
    dlclose(decrypt_handle_);
    std::cout << "Error:" << "Decrypt lib [" << lib_path
              << "] symbol find failed." << std::endl;
    return;
  }
  return;
}

void Qwen::deinit_decrypt() {
  if (lib_path.empty()) {
    return;
  }
  // Step 1: Close the dynamic library handle if it's open.
  if (decrypt_handle_) {
    dlclose(decrypt_handle_);
    decrypt_handle_ =
        nullptr; // Avoid dangling pointer by resetting to nullptr.
  }

  // Step 2: Reset the function pointer to nullptr.
  // No need to free or close anything specific for it.
  decrypt_func_ = nullptr;
}

void Qwen::init(const std::vector<int> &devices, std::string model_path) {

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
  if (!lib_path.empty()) {
    ret = bmrt_load_bmodel_with_decrypt(p_bmrt, model_path.c_str(),
                                        decrypt_func_);
  } else {
    ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  }
  assert(true == ret);
  printf("Done!\n");

  // net embed and lm_head
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
  net_penalty_sample_head =
      bmrt_get_network_info(p_bmrt, "penalty_sample_head");
  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1]; // real seqlen
  auto num_nets = bmrt_get_network_number(p_bmrt);
  if (net_greedy_head && net_penalty_sample_head) {
    lmhead_with_topk = false;
    NUM_LAYERS = (num_nets - 5) / 2;
  } else {
    lmhead_with_topk = true;
    NUM_LAYERS = (num_nets - 3) / 2;
  }

  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }
  MAX_INPUT_LENGTH = net_embed->stages[0].input_shapes[0].dims[1];
  SEQLEN = net_blocks_cache[0]->stages[0].input_shapes[3].dims[1];
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
}

void Qwen::deinit() {
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

void Qwen::net_launch(const bm_net_info_t *net, int stage_idx) {
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

void Qwen::net_launch_dyn(const bm_net_info_t *net, int stage_idx) {
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

  in_tensors[0].shape.dims[1] = TOKEN_LEN;
  in_tensors[1].shape.dims[1] = TOKEN_LEN;
  in_tensors[2].shape.dims[2] = TOKEN_LEN;
  in_tensors[2].shape.dims[3] = TOKEN_LEN;

  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

void Qwen::head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
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

int Qwen::greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];
  head_launch(net, logits_mem);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

int Qwen::penalty_sample(const bm_net_info_t *net,
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

int Qwen::forward_first(std::vector<int> &tokens) {
  std::vector<int> position_id(MAX_INPUT_LENGTH, 0);
  std::vector<uint16_t> attention_mask(MAX_INPUT_LENGTH * MAX_INPUT_LENGTH,
                                       ATTENTION_MASK);
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());

  token_length = tokens.size();
  TOKEN_LEN = tokens.size();

  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  if (is_dynamic) {
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * TOKEN_LEN + j] = 0;
      }
    }
  } else {
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * MAX_INPUT_LENGTH + j] = 0;
      }
    }
  }
  // empty
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty_net(bm_handle, net_blocks[i]);
    empty_net(bm_handle, net_blocks_cache[i]);
  }

  // forward embeding
  auto in_mem = net_embed->stages[0].input_mems[0];
  auto out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)visited_tokens.data());
  net_launch(net_embed); // prefil embedding

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
    if (is_dynamic)
      net_launch_dyn(net_blocks[idx]);
    else
      net_launch(net_blocks[idx]);
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
    token = greedy_search(net_greedy_head, lm_out_mem);
  } else if (generation_mode == "penalty_sample") {
    token = penalty_sample(net_penalty_sample_head, lm_out_mem);
  }

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

int Qwen::forward_next() {
  int cur_token = visited_tokens[token_length - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = ATTENTION_MASK;
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
                       kv_bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], token_offset, out2_mem, 0,
                       kv_bytes);
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
    token = greedy_search(net_greedy_head, lm_out_mem);
  } else if (generation_mode == "penalty_sample") {
    token = penalty_sample(net_penalty_sample_head, lm_out_mem);
  }

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

std::vector<int> Qwen::generate(std::vector<int> &history_tokens, int EOS) {
  if (history_tokens.empty()) {
    printf("Sorry: your question is empty!!\n");
    history_tokens.clear();
    return {};
  }

  // make sure token not too large
  int history_length = history_tokens.size();
  if (history_length > SEQLEN - 10) {
    history_tokens.clear();
    printf("Error: your question is too large!\n");
    return {};
  }

  std::vector<int> result_tokens;
  int token = forward_first(history_tokens);
  while (token != EOS && token_length < SEQLEN &&
         token_length <= history_length + max_new_tokens) {
    result_tokens.emplace_back(token);
    token = forward_next();
  }

  return result_tokens;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Qwen>(m, "Qwen")
      .def(pybind11::init<>())
      .def("init", &Qwen::init)
      .def("forward_first", &Qwen::forward_first)
      .def("forward_next", &Qwen::forward_next)
      .def("generate", &Qwen::generate)
      .def("deinit", &Qwen::deinit)
      .def("init_decrypt", &Qwen::init_decrypt)
      .def("deinit_decrypt", &Qwen::deinit_decrypt)
      .def_readwrite("SEQLEN", &Qwen::SEQLEN) // read SEQLEN in pipeline.py
      .def_readwrite("token_length", &Qwen::token_length)
      .def_readwrite("temperature", &Qwen::temperature)
      .def_readwrite("top_p", &Qwen::top_p)
      .def_readwrite("repeat_penalty", &Qwen::repeat_penalty)
      .def_readwrite("repeat_last_n", &Qwen::repeat_last_n)
      .def_readwrite("max_new_tokens", &Qwen::max_new_tokens)
      .def_readwrite("generation_mode", &Qwen::generation_mode)
      .def_readwrite("lib_path", &Qwen::lib_path)
      .def_readwrite("prompt_mode", &Qwen::prompt_mode);
}
