//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "memory.h"
#include "bmruntime_interface.h"
#include <getopt.h>
#include <random>
#include <map>
#include <fstream>
#include <deque>

static const int WINDOW_SIZE = 3;
static const int N_GRAM = 3;
static const int G_CANDI = 3;
static const uint16_t ATTENTION_MASK = 0xC61C; // -9984 by bfloat16

class Qwen {
public:
  void init(const std::vector<int> &devid, int eos_token_id, std::string model_path);
  void chat();
  void deinit();
  int forward_first_with_topk(std::vector<int> &tokens, std::string generation_mode = "sample");
  int forward_next_with_topk(int cur_token, std::string generation_mode = "sample");
  std::vector<int> answer(std::vector<int> history_tokens);

  std::mt19937 sgen;
  Qwen() : sgen(std::random_device()()) {};
  int sample(const std::vector<float>& probs, const std::vector<int>& tokens);
  std::vector<int> jacobi_sample(const std::vector<float>& probs, const std::vector<int>& tokens);

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  std::vector<uint16_t> make_next_mask();
  std::vector<int> make_next_pid();
  void lookahead_next(std::vector<int> &sampled_tokens);


private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
  int token_count;
  int SEQLEN;     // read from bmodel
  int NUM_LAYERS; // read from bmodel
  bool io_alone;
  int EOS;
  std::vector<std::string> history;

  // jacobi
  std::mt19937 gen;
  int step;
  int verify_num;
  int window_offset;
  int candi_offset;
  int GUESS_LEN;
  std::vector<int> my_guess;
  std::map<int, std::vector<int>> token_map;
  int past_tokens[N_GRAM][WINDOW_SIZE];
  std::vector<int> result_tokens;
  std::vector<int> verified_tokens;
  std::deque<int> candidate_tokens;
};

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

void Qwen::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void Qwen::init(const std::vector<int> &devices, int eos_token_id, std::string model_path) {
  // params
  EOS = eos_token_id;

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

  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");

  // net embed and lm_head
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1]; // real seqlen
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = (num_nets - 2) / 2;

  // jacobi
  GUESS_LEN = net_embed_cache->stages[0].input_shapes[0].dims[1];
  window_offset = G_CANDI * (N_GRAM - 1) + 1;
  candi_offset = 1; // GUESS_LEN - G_CANDI * (N_GRAM - 1);

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
  io_alone = addr_mode == 1;
  for (int i = 0; i < NUM_LAYERS; i++) {
    assert(addr_mode == net_blocks_cache[i]->addr_mode);
    if (io_alone) {
      past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
      past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    } else {
      auto ret = bm_malloc_device_byte(bm_handle, &past_key[i],
                                       net_blocks_cache[i]->max_input_bytes[3]);
      assert(BM_SUCCESS == ret);
      ret = bm_malloc_device_byte(bm_handle, &past_value[i],
                                  net_blocks_cache[i]->max_output_bytes[4]);
      assert(BM_SUCCESS == ret);
    }
  }
}

void Qwen::deinit() {
  if (false == io_alone) {
    for (int i = 0; i < NUM_LAYERS; i++) {
      bm_free_device(bm_handle, past_key[i]);
      bm_free_device(bm_handle, past_value[i]);
    }
  }
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

std::vector<int> Qwen::jacobi_sample(const std::vector<float>& probs, const std::vector<int>& tokens) {
  std::vector<int> sampled_tokens(GUESS_LEN);
  for (int i = 0; i < GUESS_LEN; i++) {
    std::discrete_distribution<> dist(probs.begin()+i*GUESS_LEN, probs.begin()+(i+1)*GUESS_LEN);
    sampled_tokens[i] = tokens[dist(sgen)+i*GUESS_LEN];
  }
  return sampled_tokens;
}

int Qwen::forward_first_with_topk(std::vector<int> &tokens, std::string generation_mode) {
  std::vector<int> input_ids(SEQLEN, 0);
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  std::copy(tokens.begin(), tokens.end(), input_ids.data());

  token_count = tokens.size();

  // Sample the tokens to generate the n-gram in the first N steps
  std::uniform_int_distribution<> distrib(0, token_count);
  int a[3] = {29973, 1, 29973};
  for (int i = 0; i < WINDOW_SIZE; ++i) {
    input_ids[token_count + i] = a[i];
  }

  int token_count_ex = token_count + WINDOW_SIZE;
  for (int i = 0; i < token_count_ex; i++) {
    position_id[i] = i;
  }
  for (int i = 0; i < token_count_ex; i++) {
    for (int j = 0; j < SEQLEN; j++) {
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

  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    d2d(in0_mem, out_mem);
    if (idx == 0) {
      // only first time need copy
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    }
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2]);
  }

  int bytes = out_mem.size / SEQLEN;
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_logits_mem = net_lm->stages[0].output_mems[0];
  auto &lm_out_tokens_mem = net_lm->stages[0].output_mems[1];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (token_count - 1) * bytes, (WINDOW_SIZE + 1) * bytes);
  net_launch(net_lm);

  // get logit & token
  int candidate_num = net_lm->stages[0].output_shapes[0].dims[1];
  std::vector<float> lm_logits(GUESS_LEN * candidate_num);
  bm_memcpy_d2s(bm_handle, lm_logits.data(), lm_out_logits_mem);
  std::vector<int> lm_tokens(GUESS_LEN * candidate_num);
  bm_memcpy_d2s(bm_handle, lm_tokens.data(), lm_out_tokens_mem);

  // select final token from candidate tokens
  auto sampled_tokens = jacobi_sample(lm_logits, lm_tokens);

  // process the lookahead tokens
  memcpy(past_tokens[0], sampled_tokens.data() + 1, WINDOW_SIZE * sizeof(int));
  return sampled_tokens[0];
}

// make attention mask for jacobi
std::vector<uint16_t> Qwen::make_next_mask() {
  int max_len_ex = SEQLEN + GUESS_LEN;
  std::vector<uint16_t> attention_mask(GUESS_LEN * max_len_ex, 0);
  for (int i = 0; i < GUESS_LEN; i++) {
    for (int j = token_count - 1; j < SEQLEN; j++) {
      attention_mask[i * max_len_ex + j] = ATTENTION_MASK;
    }
  }
  for (int i = 0; i < GUESS_LEN; i++) {
    for (int j = 1; j < GUESS_LEN; j++) {
      if (j < i) {
        // assert(candi_offset >= 1 + WINDOW_SIZE);
        if (i < window_offset) {
          int inner_offset = (i - candi_offset) % (N_GRAM - 1);
          if (j > 0 && j < i - inner_offset) {
            attention_mask[i * max_len_ex + j + SEQLEN] = ATTENTION_MASK;
          }
        } else if (j < window_offset) {
          attention_mask[i * max_len_ex + j + SEQLEN] = ATTENTION_MASK;
        }
      } else if (j > i) {
        attention_mask[i * max_len_ex + j + SEQLEN] = ATTENTION_MASK;
      }
    }
  }
  return attention_mask;
}

// make position id for jacobi
std::vector<int> Qwen::make_next_pid() {
  std::vector<int> position_id(GUESS_LEN, 0);
  position_id[0] = token_count - 1;
  for (int i = 1; i < GUESS_LEN; i++) {
    if (i <= window_offset) {
      position_id[i] = token_count + (i - 1) % (N_GRAM - 1);
    } else {
      position_id[i] = token_count + i - window_offset;
    }
  }
  return position_id;
}

int Qwen::forward_next_with_topk(int cur_token, std::string generation_mode) {
  token_count += 1;

  if (candidate_tokens.empty()) {
    auto attention_mask = make_next_mask();
    auto position_id = make_next_pid();

    // embedding
    auto &in_mem = net_embed_cache->stages[0].input_mems[0];
    auto &out_mem = net_embed_cache->stages[0].output_mems[0];
    bm_memcpy_s2d(bm_handle, in_mem, (void *)&cur_token);
    net_launch(net_embed_cache);

    // blocks
    int bytes =
        bm_mem_get_device_size(past_key[0]) / SEQLEN;
    int token_offset = (token_count - 1) * bytes;
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
      if (io_alone) {
        if (idx == 0) {
          bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
          bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
        } else {
          d2d(in1_mem, net_blocks_cache[0]->stages[0].input_mems[1]);
          d2d(in2_mem, net_blocks_cache[0]->stages[0].input_mems[2]);
        }
      } else {
        if (idx == 0) {
          bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
          bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
        }
        d2d(in3_mem, past_key[idx]);
        d2d(in4_mem, past_value[idx]);
      }
      net_launch(net_blocks_cache[idx]);
      out_mem = out0_mem;
      bm_memcpy_d2d_byte(bm_handle, past_key[idx], token_offset, out1_mem, 0,
                        bytes);
      bm_memcpy_d2d_byte(bm_handle, past_value[idx], token_offset, out2_mem, 0,
                        bytes);
    }
    auto &lm_in_mem = net_lm->stages[0].input_mems[0];
    auto &lm_out_logits_mem = net_lm->stages[0].output_mems[0];
    auto &lm_out_tokens_mem = net_lm->stages[0].output_mems[1];
    d2d(lm_in_mem, out_mem);
    net_launch(net_lm);

    int candidate_num = net_lm->stages[0].output_shapes[0].dims[1];
    std::vector<float> lm_logits(GUESS_LEN * candidate_num);
    bm_memcpy_d2s(bm_handle, lm_logits.data(), lm_out_logits_mem);
    std::vector<int> lm_tokens(GUESS_LEN * candidate_num);
    bm_memcpy_d2s(bm_handle, lm_tokens.data(), lm_out_tokens_mem);

    // select final token from candidate tokens
    auto sampled_tokens = jacobi_sample(lm_logits, lm_tokens);

    // process the lookahead tokens
    lookahead_next(sampled_tokens);
    candidate_tokens.assign(verified_tokens.begin(), verified_tokens.end());
  }

  int token = candidate_tokens[0];
  candidate_tokens.pop_front();
  return token;
}

void Qwen::lookahead_next(std::vector<int> &sampled_tokens) {
  int bytes =
      bm_mem_get_device_size(past_key[0]) / SEQLEN;

  // process the lookahead tokens
  int out_tokens[GUESS_LEN] = {0};
  std::copy(sampled_tokens.begin(), sampled_tokens.end(), out_tokens);
  if (step < N_GRAM) {
    memcpy(past_tokens[step], out_tokens + window_offset, WINDOW_SIZE * sizeof(int));
    step++;
  } else {
    for (int i = 0; i < N_GRAM - 1; i++) {
      memcpy(past_tokens[i], past_tokens[i+1], WINDOW_SIZE * sizeof(int));
    }
    memcpy(past_tokens[N_GRAM-1], out_tokens + window_offset, WINDOW_SIZE * sizeof(int));
  }
  if (step == N_GRAM) {
    for (int i = 0; i < WINDOW_SIZE; i++) {
      bool exist = false;
      if (token_map[past_tokens[0][i]].size() > 0) {
        for (int g = 0; g < G_CANDI; g++) {
          int same_num = 0;
          for (int j = 1; j <N_GRAM; j++) {
            same_num += token_map[past_tokens[0][i]][g * (N_GRAM-1) + j] == past_tokens[j][i];
          }
          if (same_num == N_GRAM - 1) {
            exist = true;
            break;
          }
        }
      }
      if (exist) {
        continue;
      }
      for (int j = 1; j < N_GRAM; j++) {
        if (token_map[past_tokens[0][i]].size() >= G_CANDI * (N_GRAM - 1)) {
          memcpy(token_map[past_tokens[0][i]].data(),
                token_map[past_tokens[0][i]].data() + N_GRAM - 1,
                (N_GRAM - 1) * (G_CANDI - 1) * sizeof(int));
          token_map[past_tokens[0][i]].resize((G_CANDI - 1) * (N_GRAM - 1));
        }
        token_map[past_tokens[0][i]].push_back(past_tokens[j][i]);
      }
    }
  }

  // verify tokens
  int max_hit = 0;
  int hit_point = 0;
  int max_hits[N_GRAM] = {0};
  if (verify_num > 0) {
    std::vector<int> correct(N_GRAM, out_tokens[0]);
    for (int i = 0; i < verify_num; i++) {
      memcpy(correct.data() + 1,
            out_tokens + 1 + i * (N_GRAM - 1),
            sizeof(int) * (N_GRAM - 1));
      int j = 0;
      for (j = 0; j < (N_GRAM - 1); j++) {
        if (correct[j] != my_guess[i * (N_GRAM - 1) + j]) {
          break;
        }
      }
      if (j > max_hit) {
        hit_point = i;
        max_hit = j;
        memcpy(max_hits, correct.data(), (max_hit+1) * sizeof(int));
      }
    }
  }

  verified_tokens.clear();
  verified_tokens.push_back(out_tokens[0]);
  if (max_hit > 0) {
    for (int i = 1; i < max_hit+1; i++) {
      verified_tokens.push_back(max_hits[i]);
    }
    // process past_keys/past_values
    int guess_offset = (1 + hit_point * (N_GRAM - 1)) * bytes;
    int token_offset = token_count * bytes;
    for (int i = 0; i < NUM_LAYERS; ++i) {
      auto &out1_mem = net_blocks_cache[i]->stages[0].output_mems[1];
      auto &out2_mem = net_blocks_cache[i]->stages[0].output_mems[2];
      bm_memcpy_d2d_byte(bm_handle, past_key[i], token_offset,
                         out1_mem, guess_offset,
                         max_hit * bytes);
      bm_memcpy_d2d_byte(bm_handle, past_value[i], token_offset,
                         out2_mem, guess_offset,
                         max_hit * bytes);
    }
  }

  // guess tokens
  // verified_tokens.resize(1);
  out_tokens[0] = verified_tokens[verified_tokens.size() - 1];
  auto it = token_map.find(out_tokens[0]);
  if (it != token_map.end()) {
    verify_num = it->second.size() / (N_GRAM - 1);
    memcpy(out_tokens+1, it->second.data(), it->second.size() * sizeof(int));
    sampled_tokens.assign(out_tokens, out_tokens + GUESS_LEN);
    my_guess = it->second;
  } else {
    if (max_hit > 0) {
      sampled_tokens.assign(out_tokens, out_tokens + GUESS_LEN);
    }
    verify_num = 0;
    my_guess.clear();
  }
}

std::vector<int> Qwen::answer(std::vector<int> history_tokens) {
  // init
  step = 0;
  verify_num = 0;
  token_map.clear();
  candidate_tokens.clear();
  result_tokens.clear();

  int token = forward_first_with_topk(history_tokens);
  while (token != EOS && token_count < SEQLEN) {
    result_tokens.push_back(token);
    token = forward_next_with_topk(token);
  }
  return result_tokens;
}

PYBIND11_MODULE(chat_jacobi, m) {
    pybind11::class_<Qwen>(m, "Qwen")
        .def(pybind11::init<>())
        .def("init", &Qwen::init)
        .def("forward_first_with_topk", &Qwen::forward_first_with_topk)
        .def("forward_next_with_topk", &Qwen::forward_next_with_topk)
        .def("answer", &Qwen::answer)
        .def("deinit", &Qwen::deinit);
}
