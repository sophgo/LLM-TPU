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
#include "memory.h"
#include "tokenizer.h"
#include "bmruntime_interface.h"
#include <getopt.h>
#include <random>
#include <map>
#include <fstream>

static const int WINDOW_SIZE = 3;
static const int N_GRAM = 3;
static const int G_CANDI = 3;
static const int GUESS_LEN = 8;
static const uint16_t ATTENTION_MASK = 0xC61C; // -9984 by bfloat16

class Qwen {
public:
  void init(const std::vector<int> &devid, std::string model_path, std::string tokenizer_path);
  void chat();
  void deinit();

  std::mt19937 sgen;
  Qwen() : sgen(std::random_device()()) {};
  int sample(const std::vector<float>& probs, const std::vector<int>& tokens);
  std::vector<int> jacobi_sample(const std::vector<float>& probs, const std::vector<int>& tokens);

private:
  void answer(const std::string &input_str);
  int forward_first(std::vector<int> &tokens);
  void forward_next();
  int forward_first_with_topk(std::vector<int> &tokens, std::string generation_mode = "sample");
  void forward_next_with_topk(int cur_token, std::string generation_mode = "sample");
  void load_tiktoken(std::string tokenizer_path);
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);


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
  std::unique_ptr<QwenTokenizer> tk;
  std::vector<std::string> history;

  // jacobi
  std::mt19937 gen;
  int step;
  int verify_num;
  std::vector<int> my_guess;
  std::map<int, std::vector<int>> token_map; // maybe warm up to speed
  int past_tokens[N_GRAM][WINDOW_SIZE];
  std::vector<int> verified_tokens;
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

void Qwen::load_tiktoken(std::string tokenizer_path) {
  printf("Load %s ... \n", tokenizer_path.c_str());
  tk = std::make_unique<QwenTokenizer>(tokenizer_path);
}

void Qwen::init(const std::vector<int> &devices, std::string model_path, std::string tokenizer_path) {
  // load tokenizer
  load_tiktoken(tokenizer_path);

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

int Qwen::forward_first(std::vector<int> &tokens) {
  std::vector<int> input_ids(SEQLEN, 0);
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  std::copy(tokens.begin(), tokens.end(), input_ids.data());

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
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (token_count - 1) * bytes, (WINDOW_SIZE + 1) * bytes);
  net_launch(net_lm);

  // process the lookahead tokens
  int out_tokens[GUESS_LEN] = {0};
  bm_memcpy_d2s(bm_handle, (void *)out_tokens, lm_out_mem);
  // int window_offset = G_CANDI * (N_GRAM s- 1) + 1;
  memcpy(past_tokens[0], out_tokens + 1, WINDOW_SIZE * sizeof(int));
  // memcpy(out_tokens + window_offset, out_tokens + 1, WINDOW_SIZE * sizeof(int));
  // bm_memcpy_s2d(bm_handle, lm_out_mem, (void *)out_tokens);
  return out_tokens[0];
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

  // process the lookahead tokens
  auto sampled_tokens = jacobi_sample(lm_logits, lm_tokens);
  memcpy(past_tokens[0], sampled_tokens.data() + 1, WINDOW_SIZE * sizeof(int));
  return sampled_tokens[0];
}

void Qwen::forward_next() {
  // make mask
  int max_len_ex = SEQLEN + GUESS_LEN;
  std::vector<uint16_t> attention_mask(GUESS_LEN * max_len_ex, 0);
  for (int i = 0; i < GUESS_LEN; i++) {
    for (int j = token_count - 1; j < SEQLEN; j++) {
      attention_mask[i * max_len_ex + j] = ATTENTION_MASK;
    }
  }
  int window_offset = G_CANDI * (N_GRAM - 1) + 1;
  int candi_offset = 1; // GUESS_LEN - G_CANDI * (N_GRAM - 1);
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

  // make position
  std::vector<int> position_id(GUESS_LEN, 0);
  position_id[0] = token_count - 1;
  for (int i = 1; i < GUESS_LEN; i++) {
    if (i <= window_offset) {
      position_id[i] = token_count + (i - 1) % (N_GRAM - 1);
    } else {
      position_id[i] = token_count + i - window_offset;
    }
  }

  // embedding
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  auto &in_mem = net_embed_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_cache->stages[0].output_mems[0];
  d2d(in_mem, lm_out_mem);
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
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm);


  // process the lookahead tokens
  int out_tokens[GUESS_LEN] = {0};
  bm_memcpy_d2s(bm_handle, (void *)out_tokens, lm_out_mem);
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
    bm_memcpy_s2d(bm_handle, lm_out_mem, (void*)out_tokens);
    my_guess = it->second;
  } else {
    if (max_hit > 0) {
      bm_memcpy_s2d(bm_handle, lm_out_mem, (void*)out_tokens);
    }
    verify_num = 0;
    my_guess.clear();
  }

  return;
}

void Qwen::forward_next_with_topk(int cur_token, std::string generation_mode) {
  // make mask
  int max_len_ex = SEQLEN + GUESS_LEN;
  std::vector<uint16_t> attention_mask(GUESS_LEN * max_len_ex, 0);
  for (int i = 0; i < GUESS_LEN; i++) {
    for (int j = token_count - 1; j < SEQLEN; j++) {
      attention_mask[i * max_len_ex + j] = ATTENTION_MASK;
    }
  }
  int window_offset = G_CANDI * (N_GRAM - 1) + 1;
  int candi_offset = 1; // GUESS_LEN - G_CANDI * (N_GRAM - 1);
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

  // make position
  std::vector<int> position_id(GUESS_LEN, 0);
  position_id[0] = token_count - 1;
  for (int i = 1; i < GUESS_LEN; i++) {
    if (i <= window_offset) {
      position_id[i] = token_count + (i - 1) % (N_GRAM - 1);
    } else {
      position_id[i] = token_count + i - window_offset;
    }
  }

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

  return;
}

void Qwen::chat() {
  while (true) {
    std::cout << "\nQuestion: ";
    std::string input_str;
    std::getline(std::cin, input_str);
    if (input_str.empty()) {
      continue;
    }
    if (input_str == "exit" || input_str == "quit") {
      break;
    }
    if (input_str == "clear") {
      history.clear();
      continue;
    }
    std::cout << "\nAnswer: " << std::flush;
    answer(input_str);
    std::cout << std::endl;
  }
}

void Qwen::answer(const std::string &input_str) {
  history.emplace_back(std::move(input_str));
  auto inp_ids = tk->encode_history(history, SEQLEN);
  token_count = inp_ids.size();

  // make sure token not too large
  if (token_count > SEQLEN - 10) {
    printf("Error: your question is too large!\n");
    return;
  }

  auto first_st = std::chrono::system_clock::now();
  int pre_token = 0;
  int token = forward_first_with_topk(inp_ids);
  auto first_et = std::chrono::system_clock::now();
  std::string pre_word;
  std::string word;
  std::vector<int> pre_ids = {pre_token};
  std::vector<int> ids = {pre_token, token};
  pre_word = tk->decode(pre_ids);
  word = tk->decode(ids);
  std::string diff = word.substr(pre_word.size());
  std::cout << diff << std::flush;

  step = 0;
  verify_num = 0;
  token_map.clear();
  std::string result;
  int tok_num = 0;
  int guessed_tok_num = 0;
  std::vector<int> tokens;
  tokens.push_back(token);
  bool end_text = false;

  auto st = std::chrono::system_clock::now();
  while (!end_text && tokens[tokens.size() - 1] != tk->im_end_id && token_count < SEQLEN) {
    forward_next_with_topk(tokens[tokens.size() - 1]);
    tokens = verified_tokens;
    tok_num += tokens.size();
    if (token_count < SEQLEN) {
      token_count += tokens.size();
      guessed_tok_num += tokens.size()-1;
    }
    uint32_t idx = std::find(tokens.begin(), tokens.end(), tk->im_end_id) - tokens.begin();
    end_text = idx < tokens.size();
    if (idx != 0) {
      std::string pre_word;
      std::string word;
      std::vector<int> pre_ids = {pre_token};
      std::vector<int> ids = {pre_token, tokens[0]};
      pre_word = tk->decode(pre_ids);
      
      word = tk->decode(ids);
      std::string diff = word.substr(pre_word.size());
      result += diff;
      std::cout << diff << std::flush;
      // print the successfully guessed tokens
      if (tokens.size() > 1 && idx > 1) {
        word = "";
        ids.resize(1);
        ids.insert(ids.begin()+1, tokens.begin()+1, tokens.begin()+idx);
        word = tk->decode(ids);
        diff = word.substr(pre_word.size());
        result += diff;
        std::cout << "\x1b[32m" << diff << "\033[0m" << std::flush;
      }
    }
  }
  auto et = std::chrono::system_clock::now();
  auto ftl_dur =
      std::chrono::duration_cast<std::chrono::microseconds>(first_et - first_st);
  auto tps_dur =
      std::chrono::duration_cast<std::chrono::microseconds>(et - st);
  if (token_count >= SEQLEN) {
    printf(" ......\nWarning: cleanup early history\n");
  }
  // double tht = tokens.size() / (tht_dur.count() * 1e-6);
  printf("\nFTL: %f s\n", (ftl_dur.count() * 1e-6));
  printf("TPS: %f token/s\n", tok_num / (tps_dur.count() * 1e-6));
  printf("generated_tokens: %d, success_guessed_tokens: %d, compression_ratio %f\n",
        tok_num, guessed_tok_num, tok_num * 1./(tok_num - guessed_tok_num));
  history.emplace_back(result);
  if (token_count + 128 >= SEQLEN) {
    int num = (history.size() + 3) / 4 * 2;
    history.erase(history.begin(), history.begin() + num);
  }
}

static void split(const std::string &s, const std::string &delim,
                  std::vector<std::string> &ret) {
  size_t last = 0;
  size_t index = s.find_first_of(delim, last);
  while (index != std::string::npos) {
    ret.push_back(s.substr(last, index - last));
    last = index + 1;
    index = s.find_first_of(delim, last);
  }
  if (last < s.length()) {
    ret.push_back(s.substr(last));
  }
}

static std::vector<int> parseCascadeDevices(const std::string &str) {
  std::vector<int> devices;
  std::vector<std::string> sub_str;
  split(str, ",", sub_str);
  for (auto &s : sub_str) {
    devices.push_back(std::atoi(s.c_str()));
  }
  return devices;
}

void Usage() {
  printf("Usage:\n"
         "  --help         : Show help info.\n"
         "  --model        : Set model path \n"
         "  --tokenizer    : Set tokenizer path \n"
         "  --devid        : Set devices to run for model, e.g. 1,2. if not "
         "set, use 0\n");
}

void processArguments(int argc, char *argv[], std::string &model_path, std::string &tokenizer_path,
                      std::vector<int> &devices) {
  struct option longOptions[] = {{"model", required_argument, nullptr, 'm'},
                                 {"tokenizer", required_argument, nullptr, 't'},
                                 {"devid", required_argument, nullptr, 'd'},
                                 {"help", no_argument, nullptr, 'h'},
                                 {nullptr, 0, nullptr, 0}};

  int optionIndex = 0;
  int option;

  while ((option = getopt_long(argc, argv, "m:t:d:h:", longOptions,
                               &optionIndex)) != -1) {
    switch (option) {
    case 'm':
      model_path = optarg;
      break;
    case 't':
      tokenizer_path = optarg;
      break;
    case 'd':
      devices = parseCascadeDevices(optarg);
      break;
    case 'h':
      Usage();
      exit(EXIT_SUCCESS);
    case '?':
      Usage();
      exit(EXIT_FAILURE);
    default:
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char **argv) {
  // set your bmodel path here
  printf("Demo for Qwen in BM1684X\n");
  std::string model_path;
  std::string tokenizer_path;
  std::vector<int> devices = {0};
  processArguments(argc, argv, model_path, tokenizer_path, devices);
  if (model_path.empty()) {
    Usage();
    exit(EXIT_FAILURE);
  }

  Qwen qwen;
  printf("Init Environment ...\n");
  qwen.init(devices, model_path, tokenizer_path);
  printf("==========================\n");
  qwen.chat();
  qwen.deinit();
  return 0;
}

