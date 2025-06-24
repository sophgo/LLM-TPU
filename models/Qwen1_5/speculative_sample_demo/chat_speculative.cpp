//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <getopt.h>
#include <inttypes.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "bmruntime_interface.h"
#include "memory.h"

static const int K = 4;
static const int GUESS_LEN = K + 1;
static const uint16_t ATTENTION_MASK = 0xF0E2;

class Qwen {
public:
  void init(const std::vector<int> &devid, std::string draft_model_path,
            std::string target_model_path);
  void deinit();
  int draft_forward_first(std::vector<int> &tokens);
  int draft_forward_next(int index);
  std::pair<std::vector<float>, std::vector<int>>
  target_forward_first(std::vector<int> &tokens);
  std::pair<std::vector<float>, std::vector<int>> target_forward_next();
  std::vector<int> generate(std::vector<int> &history_tokens, int EOS);

  std::mt19937 sgen;
  Qwen() : sgen(42){};

private:
  void net_launch(void *p_bmrt, const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);

  void head_launch(void *p_bmrt, const bm_net_info_t *net,
                   bm_device_mem_t &logits_mem);
  int greedy_search(void *p_bmrt, const bm_net_info_t *net,
                    bm_device_mem_t &logits_mem);
  std::pair<std::vector<float>, std::vector<int>>
  penalty_sample(void *p_bmrt, const bm_net_info_t *net,
                 bm_device_mem_t &logits_mem,
                 const std::vector<int> &visited_tokens, int token_length);
  std::pair<std::vector<float>, std::vector<int>> batch_penalty_sample(
      void *p_bmrt, const bm_net_info_t *net, bm_device_mem_t &logits_mem,
      const std::vector<int> &visited_tokens, int token_length);
  void roll_back(std::vector<float> &probs, std::vector<int> &tokens,
                 std::vector<float> &prob_history, int index);
  int sample_from_probs(std::vector<float> &probs, std::vector<int> &tokens);
  int verify(std::vector<int> &guess_tokens,
             std::uniform_real_distribution<float> &udist);
  int resample(std::vector<float> &probs, std::vector<int> &tokens,
               int accepted);

public:
  int SEQLEN;            // read from bmodel
  int DRAFT_NUM_LAYERS;  // read from bmodel
  int TARGET_NUM_LAYERS; // read from bmodel
  int candidate_num;     // read from bmodel
  bool draft_io_alone;
  bool target_io_alone;
  int VOCAB_SIZE;
  std::vector<int> draft_visited_tokens;
  std::vector<int> target_visited_tokens;
  int draft_token_length;
  int target_token_length;
  std::vector<float> draft_prob_history;
  std::vector<float> target_prob_history;

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

  void *d_bmrt;
  std::vector<const bm_net_info_t *> draft_net_blocks;
  std::vector<const bm_net_info_t *> draft_net_blocks_cache;
  const bm_net_info_t *draft_net_embed;
  const bm_net_info_t *draft_net_embed_cache;
  const bm_net_info_t *draft_net_lm, *draft_net_greedy_head,
      *draft_net_penalty_sample_head;
  std::vector<bm_device_mem_t> draft_past_key;
  std::vector<bm_device_mem_t> draft_past_value;

  void *t_bmrt;
  std::vector<const bm_net_info_t *> target_net_blocks;
  std::vector<const bm_net_info_t *> target_net_blocks_cache;
  const bm_net_info_t *target_net_embed;
  const bm_net_info_t *target_net_embed_cache;
  const bm_net_info_t *target_net_lm, *target_net_greedy_head,
      *target_net_penalty_sample_head;
  std::vector<bm_device_mem_t> target_past_key;
  std::vector<bm_device_mem_t> target_past_value;
};

void Qwen::net_launch(void *p_bmrt, const bm_net_info_t *net, int stage_idx) {
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

void Qwen::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void Qwen::init(const std::vector<int> &devices, std::string draft_model_path,
                std::string target_model_path) {
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
  d_bmrt = bmrt_create(handles[0]);
  t_bmrt = bmrt_create(handles[0]);
#else
  d_bmrt = bmrt_create_ex(handles.data(), handles.size());
  t_bmrt = bmrt_create_ex(handles.data(), handles.size());
#endif
  assert(NULL != d_bmrt);
  assert(NULL != t_bmrt);

  // load bmodel by file
  printf("Model[%s] loading ....\n", draft_model_path.c_str());
  assert(true == bmrt_load_bmodel(d_bmrt, draft_model_path.c_str()));

  printf("Model[%s] loading ....\n", target_model_path.c_str());
  assert(true == bmrt_load_bmodel(t_bmrt, target_model_path.c_str()));
  printf("Done!\n");

  // draft net embed and lm_head
  draft_net_embed = bmrt_get_network_info(d_bmrt, "embedding");
  draft_net_embed_cache = bmrt_get_network_info(d_bmrt, "embedding_cache");
  draft_net_lm = bmrt_get_network_info(d_bmrt, "lm_head");
  draft_net_greedy_head = bmrt_get_network_info(d_bmrt, "greedy_head");
  draft_net_penalty_sample_head =
      bmrt_get_network_info(d_bmrt, "penalty_sample_head");
  auto draft_num_nets = bmrt_get_network_number(d_bmrt);
  DRAFT_NUM_LAYERS = (draft_num_nets - 5) / 2;

  // draft net blocks
  for (int i = 0; i < DRAFT_NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    draft_net_blocks.emplace_back(
        bmrt_get_network_info(d_bmrt, block_name.c_str()));
    draft_net_blocks_cache.emplace_back(
        bmrt_get_network_info(d_bmrt, cache_name.c_str()));
  }

  // draft kv cache
  draft_past_key.resize(DRAFT_NUM_LAYERS);
  draft_past_value.resize(DRAFT_NUM_LAYERS);
  auto draft_addr_mode = draft_net_blocks_cache[0]->addr_mode;
  draft_io_alone = draft_addr_mode == 1;
  for (int i = 0; i < DRAFT_NUM_LAYERS; i++) {
    assert(draft_addr_mode == draft_net_blocks_cache[i]->addr_mode);
    if (draft_io_alone) {
      draft_past_key[i] = draft_net_blocks_cache[i]->stages[0].input_mems[3];
      draft_past_value[i] = draft_net_blocks_cache[i]->stages[0].input_mems[4];
    } else {
      auto ret =
          bm_malloc_device_byte(bm_handle, &draft_past_key[i],
                                draft_net_blocks_cache[i]->max_input_bytes[3]);
      assert(BM_SUCCESS == ret);
      ret =
          bm_malloc_device_byte(bm_handle, &draft_past_value[i],
                                draft_net_blocks_cache[i]->max_input_bytes[4]);
      assert(BM_SUCCESS == ret);
    }
  }

  // target net embed and lm_head
  target_net_embed = bmrt_get_network_info(t_bmrt, "embedding");
  target_net_embed_cache = bmrt_get_network_info(t_bmrt, "embedding_cache");
  target_net_lm = bmrt_get_network_info(t_bmrt, "lm_head");
  auto target_num_nets = bmrt_get_network_number(t_bmrt);
  TARGET_NUM_LAYERS = (target_num_nets - 3) / 2;

  // target net blocks
  for (int i = 0; i < TARGET_NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    target_net_blocks.emplace_back(
        bmrt_get_network_info(t_bmrt, block_name.c_str()));
    target_net_blocks_cache.emplace_back(
        bmrt_get_network_info(t_bmrt, cache_name.c_str()));
  }

  // target kv cache
  target_past_key.resize(TARGET_NUM_LAYERS);
  target_past_value.resize(TARGET_NUM_LAYERS);
  auto target_addr_mode = target_net_blocks_cache[0]->addr_mode;
  target_io_alone = target_addr_mode == 1;
  for (int i = 0; i < TARGET_NUM_LAYERS; i++) {
    assert(target_addr_mode == target_net_blocks_cache[i]->addr_mode);
    if (target_io_alone) {
      target_past_key[i] = target_net_blocks_cache[i]->stages[0].input_mems[3];
      target_past_value[i] =
          target_net_blocks_cache[i]->stages[0].input_mems[4];
    } else {
      auto ret =
          bm_malloc_device_byte(bm_handle, &target_past_key[i],
                                target_net_blocks_cache[i]->max_input_bytes[3]);
      assert(BM_SUCCESS == ret);
      ret =
          bm_malloc_device_byte(bm_handle, &target_past_value[i],
                                target_net_blocks_cache[i]->max_input_bytes[4]);
      assert(BM_SUCCESS == ret);
    }
  }

  // resize
  assert(draft_net_embed->stages[0].input_shapes[0].dims[1] ==
         target_net_embed->stages[0].input_shapes[0].dims[1]);
  SEQLEN = draft_net_embed->stages[0].input_shapes[0].dims[1];

  VOCAB_SIZE = draft_net_lm->stages[0].output_shapes[0].dims[1];

  candidate_num =
      draft_net_penalty_sample_head->stages[0].output_shapes[0].dims[1];

  draft_visited_tokens.resize(SEQLEN);
  target_visited_tokens.resize(SEQLEN);
  draft_prob_history.resize(K * VOCAB_SIZE);
  target_prob_history.resize(K * VOCAB_SIZE);
}

void Qwen::deinit() {
  if (false == draft_io_alone) {
    for (int i = 0; i < DRAFT_NUM_LAYERS; i++) {
      bm_free_device(bm_handle, draft_past_key[i]);
      bm_free_device(bm_handle, draft_past_value[i]);
    }
  }
  if (false == target_io_alone) {
    for (int i = 0; i < TARGET_NUM_LAYERS; i++) {
      bm_free_device(bm_handle, target_past_key[i]);
      bm_free_device(bm_handle, target_past_value[i]);
    }
  }
  bmrt_destroy(d_bmrt);
  bmrt_destroy(t_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

void Qwen::head_launch(void *p_bmrt, const bm_net_info_t *net,
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

int Qwen::greedy_search(void *p_bmrt, const bm_net_info_t *net,
                        bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];
  head_launch(p_bmrt, net, logits_mem);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

std::pair<std::vector<float>, std::vector<int>>
Qwen::penalty_sample(void *p_bmrt, const bm_net_info_t *net,
                     bm_device_mem_t &logits_mem,
                     const std::vector<int> &visited_tokens, int token_length) {
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
  head_launch(p_bmrt, net, logits_mem);

  // get logit & token
  int candidate_num = net->stages[0].output_shapes[0].dims[1];
  std::vector<float> probs(candidate_num);
  bm_memcpy_d2s(bm_handle, probs.data(), out0_mem);
  std::vector<int> tokens(candidate_num);
  bm_memcpy_d2s(bm_handle, tokens.data(), out1_mem);

  return std::make_pair(probs, tokens);
}

std::pair<std::vector<float>, std::vector<int>> Qwen::batch_penalty_sample(
    void *p_bmrt, const bm_net_info_t *net, bm_device_mem_t &logits_mem,
    const std::vector<int> &visited_tokens, int token_length) {
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
  head_launch(p_bmrt, net, logits_mem);

  // get logit & token
  std::vector<float> probs(candidate_num * GUESS_LEN);
  bm_memcpy_d2s(bm_handle, probs.data(), out0_mem);
  std::vector<int> tokens(candidate_num * GUESS_LEN);
  bm_memcpy_d2s(bm_handle, tokens.data(), out1_mem);

  return std::make_pair(probs, tokens);
}

void Qwen::roll_back(std::vector<float> &probs, std::vector<int> &tokens,
                     std::vector<float> &prob_history, int index) {
  for (size_t i = 0; i < tokens.size(); i++) {
    prob_history[tokens[i] + index * VOCAB_SIZE] = probs[i];
  }
}

int Qwen::sample_from_probs(std::vector<float> &probs,
                            std::vector<int> &tokens) {
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return tokens[dist(sgen)];
}

//===------------------------------------------------------------===//
// Draft Model Forward
//===------------------------------------------------------------===//

int Qwen::draft_forward_first(std::vector<int> &tokens) {
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  std::copy(tokens.begin(), tokens.end(), draft_visited_tokens.data());

  draft_token_length = tokens.size();

  for (int i = 0; i < draft_token_length; i++) {
    position_id[i] = i;
  }
  for (int i = 0; i < draft_token_length; i++) {
    for (int j = 0; j < SEQLEN; j++) {
      if (j <= i) {
        attention_mask[i * SEQLEN + j] = 0;
      }
    }
  }

  // forward embeding
  auto &in_mem = draft_net_embed->stages[0].input_mems[0];
  auto &out_mem = draft_net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)draft_visited_tokens.data());
  net_launch(d_bmrt, draft_net_embed); // prefil embedding

  // forward blocks
  for (int idx = 0; idx < DRAFT_NUM_LAYERS; idx++) {
    auto &in0_mem = draft_net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = draft_net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = draft_net_blocks[idx]->stages[0].input_mems[2];
    d2d(in0_mem, out_mem);
    if (idx == 0) {
      // only first time need copy
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    }
    net_launch(d_bmrt, draft_net_blocks[idx]);
    out_mem = draft_net_blocks[idx]->stages[0].output_mems[0];
    d2d(draft_past_key[idx], draft_net_blocks[idx]->stages[0].output_mems[1]);
    d2d(draft_past_value[idx], draft_net_blocks[idx]->stages[0].output_mems[2]);
  }

  // forward lmhead
  int bytes = out_mem.size / SEQLEN;
  auto &lm_in_mem = draft_net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = draft_net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (draft_token_length - 1) * bytes, bytes);
  net_launch(d_bmrt, draft_net_lm);

  auto pair = penalty_sample(d_bmrt, draft_net_penalty_sample_head, lm_out_mem,
                             draft_visited_tokens, draft_token_length);

  auto &candidate_probs = pair.first;
  auto &candidate_tokens = pair.second;

  // roll back
  roll_back(candidate_probs, candidate_tokens, draft_prob_history, 0);

  auto token = sample_from_probs(candidate_probs, candidate_tokens);
  draft_visited_tokens[draft_token_length] = token;
  draft_token_length += 1;
  return token;
}

int Qwen::draft_forward_next(int index) {
  int cur_token = draft_visited_tokens[draft_token_length - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = draft_token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = draft_token_length - 1;
  // embedding
  auto &in_mem = draft_net_embed_cache->stages[0].input_mems[0];
  auto &out_mem = draft_net_embed_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)&cur_token);
  net_launch(d_bmrt, draft_net_embed_cache);

  // blocks
  int bytes = bm_mem_get_device_size(
      draft_net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (draft_token_length - 1) * bytes;
  for (int idx = 0; idx < DRAFT_NUM_LAYERS; idx++) {
    auto &in0_mem = draft_net_blocks_cache[idx]->stages[0].input_mems[0];
    auto &in1_mem = draft_net_blocks_cache[idx]->stages[0].input_mems[1];
    auto &in2_mem = draft_net_blocks_cache[idx]->stages[0].input_mems[2];
    auto &in3_mem = draft_net_blocks_cache[idx]->stages[0].input_mems[3];
    auto &in4_mem = draft_net_blocks_cache[idx]->stages[0].input_mems[4];
    auto &out0_mem = draft_net_blocks_cache[idx]->stages[0].output_mems[0];
    auto &out1_mem = draft_net_blocks_cache[idx]->stages[0].output_mems[1];
    auto &out2_mem = draft_net_blocks_cache[idx]->stages[0].output_mems[2];
    d2d(in0_mem, out_mem);
    if (draft_io_alone) {
      if (idx == 0) {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      } else {
        d2d(in1_mem, draft_net_blocks_cache[0]->stages[0].input_mems[1]);
        d2d(in2_mem, draft_net_blocks_cache[0]->stages[0].input_mems[2]);
      }
    } else {
      if (idx == 0) {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      }
      d2d(in3_mem, draft_past_key[idx]);
      d2d(in4_mem, draft_past_value[idx]);
    }
    net_launch(d_bmrt, draft_net_blocks_cache[idx]);
    out_mem = out0_mem;
    bm_memcpy_d2d_byte(bm_handle, draft_past_key[idx], token_offset, out1_mem,
                       0, bytes);
    bm_memcpy_d2d_byte(bm_handle, draft_past_value[idx], token_offset, out2_mem,
                       0, bytes);
  }

  // forward lmhead
  auto &lm_in_mem = draft_net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = draft_net_lm->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(d_bmrt, draft_net_lm);

  auto pair = penalty_sample(d_bmrt, draft_net_penalty_sample_head, lm_out_mem,
                             draft_visited_tokens, draft_token_length);

  auto &candidate_probs = pair.first;
  auto &candidate_tokens = pair.second;

  // roll back
  roll_back(candidate_probs, candidate_tokens, draft_prob_history, index);

  auto token = sample_from_probs(candidate_probs, candidate_tokens);
  draft_visited_tokens[draft_token_length] = token;
  draft_token_length += 1;
  return token;
}

//===------------------------------------------------------------===//
// Target Model Forward
//===------------------------------------------------------------===//

std::pair<std::vector<float>, std::vector<int>>
Qwen::target_forward_first(std::vector<int> &tokens) {
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  std::copy(tokens.begin(), tokens.end(), target_visited_tokens.data());

  target_token_length = tokens.size();

  for (int i = 0; i < target_token_length; i++) {
    position_id[i] = i;
  }
  for (int i = 0; i < target_token_length; i++) {
    for (int j = 0; j < SEQLEN; j++) {
      if (j <= i) {
        attention_mask[i * SEQLEN + j] = 0;
      }
    }
  }

  // forward embeding
  auto &in_mem = target_net_embed->stages[0].input_mems[0];
  auto &out_mem = target_net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)target_visited_tokens.data());
  net_launch(t_bmrt, target_net_embed); // prefil embedding

  // forward blocks
  for (int idx = 0; idx < TARGET_NUM_LAYERS; idx++) {
    auto &in0_mem = target_net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = target_net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = target_net_blocks[idx]->stages[0].input_mems[2];
    d2d(in0_mem, out_mem);
    if (idx == 0) {
      // only first time need copy
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    }
    net_launch(t_bmrt, target_net_blocks[idx]);
    out_mem = target_net_blocks[idx]->stages[0].output_mems[0];
    d2d(target_past_key[idx], target_net_blocks[idx]->stages[0].output_mems[1]);
    d2d(target_past_value[idx],
        target_net_blocks[idx]->stages[0].output_mems[2]);
  }

  // forward lmhead
  int bytes = out_mem.size / SEQLEN;
  auto &lm_in0_mem = target_net_lm->stages[0].input_mems[0];
  auto &lm_in1_mem = target_net_lm->stages[0].input_mems[1];
  auto &lm_in2_mem = target_net_lm->stages[0].input_mems[2];
  auto &lm_in3_mem = target_net_lm->stages[0].input_mems[3];
  auto &lm_in4_mem = target_net_lm->stages[0].input_mems[4];
  auto &lm_out0_mem = target_net_lm->stages[0].output_mems[0];
  auto &lm_out1_mem = target_net_lm->stages[0].output_mems[1];

  // repeat_penalty + top_p + top_k + temperature
  bm_memcpy_d2d_byte(bm_handle, lm_in0_mem, 0, out_mem,
                     (target_token_length - GUESS_LEN) * bytes,
                     GUESS_LEN * bytes);
  std::vector<int> generated_tokens(SEQLEN, target_visited_tokens[target_token_length - 1]);
  repeat_last_n = std::min(repeat_last_n, target_token_length);
  std::copy(target_visited_tokens.begin() + target_token_length - repeat_last_n,
            target_visited_tokens.begin() + target_token_length, generated_tokens.begin());
  bm_memcpy_s2d(bm_handle, lm_in1_mem, (void *)generated_tokens.data());
  bm_memcpy_s2d(bm_handle, lm_in2_mem, (void *)&top_p);
  bm_memcpy_s2d(bm_handle, lm_in3_mem, (void *)&temperature);
  bm_memcpy_s2d(bm_handle, lm_in4_mem, (void *)&repeat_penalty);

  // inference
  net_launch(t_bmrt, target_net_lm);

  // get logit & token
  std::vector<float> batch_probs(candidate_num * GUESS_LEN);
  bm_memcpy_d2s(bm_handle, batch_probs.data(), lm_out0_mem);
  std::vector<int> batch_tokens(candidate_num * GUESS_LEN);
  bm_memcpy_d2s(bm_handle, batch_tokens.data(), lm_out1_mem);


  for (int i = 0; i < K; i++) {
    std::vector<float> candidate_probs(batch_probs.begin() + i * candidate_num,
                                       batch_probs.begin() +
                                           (i + 1) * candidate_num);
    std::vector<int> candidate_tokens(batch_tokens.begin() + i * candidate_num,
                                      batch_tokens.begin() +
                                          (i + 1) * candidate_num);

    roll_back(candidate_probs, candidate_tokens, target_prob_history, i);
  }

  target_token_length += 1;
  return std::make_pair(batch_probs, batch_tokens);
}

std::pair<std::vector<float>, std::vector<int>> Qwen::target_forward_next() {
  std::vector<int> cur_tokens(
      target_visited_tokens.begin() + target_token_length - GUESS_LEN,
      target_visited_tokens.begin() + target_token_length);
  std::vector<uint16_t> attention_mask((GUESS_LEN) * (SEQLEN + GUESS_LEN),
                                       ATTENTION_MASK);
  std::vector<int> position_ids(GUESS_LEN, 0);

  for (int i = 0; i < GUESS_LEN; i++) {
    for (int j = 0; j < target_token_length - GUESS_LEN; j++) {
      attention_mask[i * (SEQLEN + GUESS_LEN) + j] = 0;
    }
    for (int j = SEQLEN; j < SEQLEN + i + 1; j++) {
      attention_mask[i * (SEQLEN + GUESS_LEN) + j] = 0;
    }
    position_ids[i] = target_token_length + i - GUESS_LEN;
  }

  // embedding
  auto &in_mem = target_net_embed_cache->stages[0].input_mems[0];
  auto &out_mem = target_net_embed_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)cur_tokens.data());
  net_launch(t_bmrt, target_net_embed_cache);

  // blocks
  int bytes = bm_mem_get_device_size(
                  target_net_blocks_cache[0]->stages[0].output_mems[1]) /
              GUESS_LEN;
  int token_offset = (target_token_length - GUESS_LEN) * bytes;
  for (int idx = 0; idx < TARGET_NUM_LAYERS; idx++) {
    auto &in0_mem = target_net_blocks_cache[idx]->stages[0].input_mems[0];
    auto &in1_mem = target_net_blocks_cache[idx]->stages[0].input_mems[1];
    auto &in2_mem = target_net_blocks_cache[idx]->stages[0].input_mems[2];
    auto &in3_mem = target_net_blocks_cache[idx]->stages[0].input_mems[3];
    auto &in4_mem = target_net_blocks_cache[idx]->stages[0].input_mems[4];
    auto &out0_mem = target_net_blocks_cache[idx]->stages[0].output_mems[0];
    auto &out1_mem = target_net_blocks_cache[idx]->stages[0].output_mems[1];
    auto &out2_mem = target_net_blocks_cache[idx]->stages[0].output_mems[2];
    d2d(in0_mem, out_mem);
    if (target_io_alone) {
      if (idx == 0) {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_ids.data());
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      } else {
        d2d(in1_mem, target_net_blocks_cache[0]->stages[0].input_mems[1]);
        d2d(in2_mem, target_net_blocks_cache[0]->stages[0].input_mems[2]);
      }
    } else {
      if (idx == 0) {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_ids.data());
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      }
      d2d(in3_mem, target_past_key[idx]);
      d2d(in4_mem, target_past_value[idx]);
    }
    net_launch(t_bmrt, target_net_blocks_cache[idx]);
    out_mem = out0_mem;
    bm_memcpy_d2d_byte(bm_handle, target_past_key[idx], token_offset, out1_mem,
                       0, GUESS_LEN * bytes);
    bm_memcpy_d2d_byte(bm_handle, target_past_value[idx], token_offset,
                       out2_mem, 0, GUESS_LEN * bytes);
  }

  // forward lmhead
  auto &lm_in0_mem = target_net_lm->stages[0].input_mems[0];
  auto &lm_in1_mem = target_net_lm->stages[0].input_mems[1];
  auto &lm_in2_mem = target_net_lm->stages[0].input_mems[2];
  auto &lm_in3_mem = target_net_lm->stages[0].input_mems[3];
  auto &lm_in4_mem = target_net_lm->stages[0].input_mems[4];
  auto &lm_out0_mem = target_net_lm->stages[0].output_mems[0];
  auto &lm_out1_mem = target_net_lm->stages[0].output_mems[1];

  // repeat_penalty + top_p + top_k + temperature
  d2d(lm_in0_mem, out_mem);
  std::vector<int> generated_tokens(SEQLEN, target_visited_tokens[target_token_length - 1]);
  repeat_last_n = std::min(repeat_last_n, target_token_length);
  std::copy(target_visited_tokens.begin() + target_token_length - repeat_last_n,
            target_visited_tokens.begin() + target_token_length, generated_tokens.begin());
  bm_memcpy_s2d(bm_handle, lm_in1_mem, (void *)generated_tokens.data());
  bm_memcpy_s2d(bm_handle, lm_in2_mem, (void *)&top_p);
  bm_memcpy_s2d(bm_handle, lm_in3_mem, (void *)&temperature);
  bm_memcpy_s2d(bm_handle, lm_in4_mem, (void *)&repeat_penalty);

  // inference
  net_launch(t_bmrt, target_net_lm);

  // get logit & token
  std::vector<float> batch_probs(candidate_num * GUESS_LEN);
  bm_memcpy_d2s(bm_handle, batch_probs.data(), lm_out0_mem);
  std::vector<int> batch_tokens(candidate_num * GUESS_LEN);
  bm_memcpy_d2s(bm_handle, batch_tokens.data(), lm_out1_mem);


  for (int i = 0; i < K; i++) {
    std::vector<float> candidate_probs(batch_probs.begin() + i * candidate_num,
                                       batch_probs.begin() +
                                           (i + 1) * candidate_num);
    std::vector<int> candidate_tokens(batch_tokens.begin() + i * candidate_num,
                                      batch_tokens.begin() +
                                          (i + 1) * candidate_num);

    roll_back(candidate_probs, candidate_tokens, target_prob_history, i);
  }

  target_token_length += 1;
  return std::make_pair(batch_probs, batch_tokens);
}

int Qwen::verify(std::vector<int> &guess_tokens,
                 std::uniform_real_distribution<float> &udist) {
  int accepted = 0;
  for (size_t i = 0; i < K; i++) {
    float randomValue = udist(sgen);
    if (randomValue >
        target_prob_history[guess_tokens[i] + VOCAB_SIZE * i] /
            draft_prob_history[guess_tokens[i] + VOCAB_SIZE * i]) {
      break;
    }
    accepted += 1;
  }
  return accepted;
}

int Qwen::resample(std::vector<float> &probs, std::vector<int> &tokens,
                   int accepted) {
  std::vector<float> modified_probs(candidate_num, 0);
  std::vector<int> modified_tokens(tokens.begin() + accepted * candidate_num,
                                   tokens.begin() +
                                       (accepted + 1) * candidate_num);
  if (accepted == K) {
    for (int i = 0; i < candidate_num; i++) {
      modified_probs[i] =
          probs[accepted * candidate_num + i] -
          draft_prob_history[tokens[accepted * candidate_num + i] +
                             accepted * VOCAB_SIZE];
    }
    draft_forward_next(0); // important !!!
  } else {
    std::copy(probs.begin() + accepted * candidate_num,
              probs.begin() + (accepted + 1) * candidate_num,
              modified_probs.begin());
  }
  return sample_from_probs(modified_probs, modified_tokens);
}

std::vector<int> Qwen::generate(std::vector<int> &history_tokens, int EOS) {
  if (history_tokens.empty()) {
    printf("Sorry: your question is empty!!\n");
    history_tokens.clear();
    return {};
  }

  // make sure token not too large
  if ((int)history_tokens.size() > SEQLEN - 10) {
    history_tokens.clear();
    printf("Error: your question is too large!\n");
    return {};
  }

  int accepted = 0;
  std::vector<int> guess_tokens;
  std::vector<int> result_tokens;
  std::uniform_real_distribution<float> udist(0.0f, 1.0f);

  // 1. Prefill
  // draft_model forward K
  guess_tokens.emplace_back(draft_forward_first(history_tokens));
  for (int i = 1; i < K; i++) {
    guess_tokens.emplace_back(draft_forward_next(i));
  }

  // target_model forward
  std::vector<int> target_tokens(history_tokens);
  target_tokens.insert(target_tokens.end(), guess_tokens.begin(),
                       guess_tokens.end());
  auto pair = target_forward_first(target_tokens);

  // Verify
  accepted = verify(guess_tokens, udist);

  for (int i = 0; i < accepted; i++) {
    result_tokens.emplace_back(guess_tokens[i]);
  }

  // Resample
  int last_token = resample(pair.first, pair.second, accepted);
  result_tokens.emplace_back(last_token);

  // 2. Decode
  while (std::find(result_tokens.end() - GUESS_LEN, result_tokens.end(), EOS) ==
             result_tokens.end() &&
         result_tokens.size() < SEQLEN - history_tokens.size() - 10) {
    guess_tokens.clear();
    draft_prob_history.clear();
    target_prob_history.clear();

    // draft model forward
    draft_token_length = history_tokens.size() + result_tokens.size();
    draft_visited_tokens[draft_token_length - 1] = last_token;
    for (int i = 0; i < K; i++) {
      guess_tokens.emplace_back(draft_forward_next(i));
    }

    // target model forward
    target_token_length = draft_token_length;
    target_visited_tokens = draft_visited_tokens;
    pair = target_forward_next();

    // verfiy
    accepted = verify(guess_tokens, udist);
    for (int i = 0; i < accepted; i++) {
      result_tokens.emplace_back(guess_tokens[i]);
    }

    // resample
    last_token = resample(pair.first, pair.second, accepted);
    result_tokens.emplace_back(last_token);
  }

  return result_tokens;
}

PYBIND11_MODULE(chat_speculative, m) {
  pybind11::class_<Qwen>(m, "Qwen")
      .def(pybind11::init<>())
      .def("init", &Qwen::init)
      // .def("forward_first", &Qwen::forward_first)
      // .def("forward_next", &Qwen::forward_next)
      .def("generate", &Qwen::generate)
      .def("deinit", &Qwen::deinit)
      .def_readwrite("SEQLEN", &Qwen::SEQLEN) // read SEQLEN in pipeline.py
      // .def_readwrite("token_length", &Qwen::token_length)
      .def_readwrite("temperature", &Qwen::temperature)
      .def_readwrite("top_p", &Qwen::top_p)
      .def_readwrite("repeat_penalty", &Qwen::repeat_penalty)
      .def_readwrite("repeat_last_n", &Qwen::repeat_last_n)
      .def_readwrite("max_new_tokens", &Qwen::max_new_tokens)
      .def_readwrite("generation_mode", &Qwen::generation_mode)
      .def_readwrite("prompt_mode", &Qwen::prompt_mode);
}
