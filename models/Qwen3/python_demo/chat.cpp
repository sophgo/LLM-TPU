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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdio.h>
#include <vector>

static const uint16_t ATTENTION_MASK = 0xC61C;

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
  int value = 0;
  for (int i = 0; i < net->input_num; i++) {
    bm_memset_device_ext(bm_handle, &value, 1,
                         net->stages[stage_idx].input_mems[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bm_memset_device_ext(bm_handle, &value, 1,
                         net->stages[stage_idx].output_mems[i]);
  }
}

class Qwen {
public:
  void init(const std::vector<int> &devid, std::string model_path);
  void deinit();
  int forward_first(std::vector<int> &tokens);
  int forward_next();
  void clear_kv();
  std::vector<int> generate(std::vector<int> &history_tokens, int EOS);

  std::mt19937 sgen;
  Qwen() : sgen(std::random_device()()) {};

private:
  int forward_first_with_kv(std::vector<int> &tokens);
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  void net_launch_dyn(const bm_net_info_t *net, int real_len,
                      int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset = 0,
                  int size = 0);
  void init_by_names();
  int greedy_search(bm_device_mem_t &logits_mem);
  int penalty_sample(bm_device_mem_t &logits_mem);

public:
  int hidden_bytes;
  int kv_bytes;
  int token_length;
  int SEQLEN;
  int MAX_INPUT_LENGTH;
  int PREFILL_KV_LENGTH;
  int NUM_LAYERS;
  bool lmhead_with_topk;
  bool is_dynamic;
  std::vector<int> visited_tokens;
  bool support_prefill_kv;
  int history_length;

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
  const bm_net_info_t *net_lm, *net_greedy_head, *net_sample_head;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};

void Qwen::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
               int size) {
  if (!size)
    size = bm_mem_get_device_size(src);
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void Qwen::init_by_names() {
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
  MAX_INPUT_LENGTH = net_embed->stages[0].input_shapes[0].dims[1];
  SEQLEN = net_blocks_cache[0]->stages[0].input_shapes[3].dims[1];
  support_prefill_kv = net_blocks[0]->input_num == 5; // with kv cache
  history_length = 0;
  printf("Num Layers:%d\n", NUM_LAYERS);
  if (support_prefill_kv) {
    PREFILL_KV_LENGTH = net_blocks[0]->stages[0].input_shapes[3].dims[1];
    printf("History by kv: True\n");
  }
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
  // load bmodel
  std::cout << "Model [" << model_path.c_str() << "] loading .... ";
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  std::cout << "Done!" << std::endl;

  init_by_names();

  // resize
  visited_tokens.resize(SEQLEN);

  hidden_bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[0]);
  kv_bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);

  auto buffer_size =
      bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  is_dynamic = net_blocks[0]->is_dynamic;
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
}

void Qwen::deinit() {
  bm_free_device(bm_handle, dev_buffer);
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

void Qwen::net_launch_dyn(const bm_net_info_t *net, int real_len,
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
  bm_thread_sync(bm_handle);
}

int Qwen::greedy_search(bm_device_mem_t &logits_mem) {
  auto &in_mem = net_greedy_head->stages[0].input_mems[0];
  auto &out_mem = net_greedy_head->stages[0].output_mems[0];
  d2d(in_mem, logits_mem, 0, bm_mem_get_device_size(logits_mem));
  net_launch(net_greedy_head);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

int Qwen::penalty_sample(bm_device_mem_t &logits_mem) {
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

int Qwen::forward_first(std::vector<int> &tokens) {
  if (support_prefill_kv) {
    return forward_first_with_kv(tokens);
  }
  std::vector<int> position_id(MAX_INPUT_LENGTH, 0);
  std::vector<uint16_t> attention_mask(MAX_INPUT_LENGTH * MAX_INPUT_LENGTH,
                                       ATTENTION_MASK);
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());

  token_length = tokens.size();

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
        attention_mask[i * MAX_INPUT_LENGTH + j] = 0;
      }
    }
  }

  // forward embeding
  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  empty(bm_handle, in_mem);
  bm_memcpy_s2d_partial(bm_handle, in_mem, (void *)tokens.data(),
                        token_length * sizeof(int));
  net_launch(net_embed);
  d2d(dev_buffer, out_mem, 0, bm_mem_get_device_size(out_mem));
  out_mem = dev_buffer;

  // forward blocks
  empty_net(bm_handle, net_blocks[0]);
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
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
  history_length = token_length;
  return token;
}

int Qwen::forward_first_with_kv(std::vector<int> &inputs) {
  int max_kv_length = MAX_INPUT_LENGTH + PREFILL_KV_LENGTH;
  std::vector<int> position_id(MAX_INPUT_LENGTH, 0);
  std::copy(inputs.begin(), inputs.end(), visited_tokens.data());
  auto old_length = history_length;
  token_length = inputs.size();
  history_length += token_length;
  std::vector<uint16_t> attention_mask(MAX_INPUT_LENGTH * max_kv_length,
                                       ATTENTION_MASK);
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
  for (int i = 0; i < token_length; i++) {
    position_id[i] = i + old_length;
  }
  // forward embeding
  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  empty(bm_handle, in_mem);
  bm_memcpy_s2d_partial(bm_handle, in_mem, (void *)inputs.data(),
                        token_length * sizeof(int));
  net_launch(net_embed);
  d2d(dev_buffer, out_mem, 0, bm_mem_get_device_size(out_mem));
  out_mem = dev_buffer;

  // forward blocks
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
                         kv_bytes * old_length);
      bm_memcpy_d2d_byte(bm_handle, in4_mem, 0, past_value[idx], 0,
                         kv_bytes * old_length);
    } else if (idx == 0) {
      empty(bm_handle, in3_mem);
      empty(bm_handle, in4_mem);
    }
    bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
    bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    auto &out1_mem = net_blocks[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks[idx]->stages[0].output_mems[2];
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], old_length * kv_bytes,
                       out1_mem, 0, kv_bytes * token_length);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], old_length * kv_bytes,
                       out2_mem, 0, kv_bytes * token_length);
  }

  // forward lmhead
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (token_length - 1) * hidden_bytes, hidden_bytes);
  net_launch(net_lm);
  int token = 0;
  if (!net_greedy_head && !net_sample_head) {
    bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  } else if (generation_mode == "greedy") {
    token = greedy_search(lm_out_mem);
  } else if (generation_mode == "sample") {
    token = penalty_sample(lm_out_mem);
  }
  visited_tokens[token_length] = token;
  token_length++;
  history_length++;
  return token;
}

int Qwen::forward_next() {
  int cur_token = visited_tokens[token_length - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = history_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = history_length - 1;
  // embedding
  auto &in_mem = net_embed_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_cache->stages[0].output_mems[0];
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
    token = greedy_search(lm_out_mem);
  } else if (generation_mode == "sample") {
    token = penalty_sample(lm_out_mem);
  }

  visited_tokens[token_length] = token;
  token_length++;
  history_length++;
  return token;
}

void Qwen::clear_kv() {
  if (!support_prefill_kv) {
    return;
  }
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  history_length = 0;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Qwen>(m, "Qwen")
      .def(pybind11::init<>())
      .def("init", &Qwen::init)
      .def("forward_first", &Qwen::forward_first)
      .def("forward_next", &Qwen::forward_next)
      .def("clear_kv", &Qwen::clear_kv)
      .def("deinit", &Qwen::deinit)
      .def_readonly("SEQLEN", &Qwen::SEQLEN) // read SEQLEN in pipeline.py
      .def_readonly("MAX_INPUT_LENGTH", &Qwen::MAX_INPUT_LENGTH)
      .def_readonly("token_length", &Qwen::token_length)
      .def_readonly("history_length", &Qwen::history_length)
      .def_readonly("support_prefill_kv", &Qwen::support_prefill_kv)
      .def_readwrite("generation_mode", &Qwen::generation_mode)
      .def_readwrite("penalty", &Qwen::penalty)
      .def_readwrite("temperature", &Qwen::temperature)
      .def_readwrite("top_k", &Qwen::top_k)
      .def_readwrite("top_p", &Qwen::top_p);
}
