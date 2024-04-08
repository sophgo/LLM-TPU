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
#include "tokenizer.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <vector>


static const uint16_t ATTENTION_MASK = 0xC61C; // -9984 by bfloat16

typedef union {
  float fval;
  uint32_t bits;
  struct {
    uint32_t frac : 23; // mantissa
    uint32_t exp : 8;   // exponent
    uint32_t sign : 1;  // sign
  } format;
} fp32;

// from chatgpt
static inline uint16_t fp32_to_bf16_bits(uint32_t f) {
  /*
   * Extract the sign of the input number into the high bit of the 16-bit word:
   *
   *      +---+-----+-------------------+
   *      | S | EEEE EEEE | MMM MMMM     |
   *      +---+-----+-------------------+
   * Bits  15  14-7          6-0
   */
  const uint32_t sign = f & UINT32_C(0x80000000);
  /*
   * Extract the exponent and the top 7 bits of the mantissa into the bits 0-14
   * of the 16-bit word:
   *
   *      +---+-----+-------------------+
   *      | 0 | EEEE EEEE | MMM MMMM     |
   *      +---+-----+-------------------+
   * Bits  14  7-0          6-0
   */
  const uint32_t rest = (f >> 16) & UINT32_C(0x7FFF);

  // Combine the sign with the rest of the number
  const uint16_t bfloat16 = (sign >> 16) | rest;

  // Handle rounding by examining the bits that are being truncated
  const uint32_t rounding_mask = UINT32_C(0x00007FFF);
  const uint32_t rounding_bits = f & rounding_mask;
  const uint32_t halfway = UINT32_C(0x00004000);
  if (rounding_bits > halfway || (rounding_bits == halfway && (bfloat16 & 1))) {
    // Round up
    return bfloat16 + 1;
  } else {
    // Truncate
    return bfloat16;
  }
}

static inline uint16_t fp16_ieee_to_fp32_value(float val) {
  fp32 f0;
  f0.fval = val;
  return fp32_to_bf16_bits(f0.bits);
}

class Qwen {
public:
  void init(const std::vector<int> &devid, std::string model_path,
            std::string tokenizer_path, const float &__temperature,
            const float &__top_p, const float &repeat_penalty,
            const int &repeat_last_n, const int &__max_new_tokens,
            const std::string &__generation_mode,
            const std::string &__input_mode);
  void deinit();
  void chat();
  void answer(const std::string &input_str);
  int forward_first(std::vector<int> &tokens);
  int forward_next();
  int token_count;
  std::vector<int> generate(std::vector<int> &history_tokens, int EOS);
  std::mt19937 sgen;
  Qwen() : sgen(std::random_device()()){};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  void load_tiktoken(std::string tokenizer_path);
  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem);

public:
  int token_length;
  int SEQLEN;     // read from bmodel
  int NUM_LAYERS; // read from bmodel
  bool io_alone;
  std::unique_ptr<QwenTokenizer> tk;
  std::vector<int> visited_tokens;
  std::vector<std::string> history;

  // generation
  float temperature;
  uint16_t top_p;
  float repeat_penalty;
  int repeat_last_n;
  int max_new_tokens;
  std::string generation_mode;
  std::string input_mode;

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

void Qwen::init(const std::vector<int> &devices, std::string model_path,
                std::string tokenizer_path, const float &__temperature,
                const float &__top_p, const float &__repeat_penalty,
                const int &__repeat_last_n, const int &__max_new_tokens,
                const std::string &__generation_mode,
                const std::string &__input_mode) {
  // load tokenizer
  load_tiktoken(tokenizer_path);

  // generation params
  temperature = __temperature;
  top_p = fp32_to_bf16_bits(__top_p);
  repeat_penalty = __repeat_penalty;
  repeat_last_n = __repeat_last_n;
  max_new_tokens = __max_new_tokens;
  generation_mode = __generation_mode;
  input_mode = __input_mode;

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
  net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
  net_penalty_sample_head = bmrt_get_network_info(p_bmrt, "penalty_sample_head");
  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1]; // real seqlen
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = (num_nets - 5) / 2;

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
                                  net_blocks_cache[i]->max_input_bytes[4]);
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

void Qwen::load_tiktoken(std::string tokenizer_path) {
  printf("Load %s ... \n", tokenizer_path.c_str());
  tk = std::make_unique<QwenTokenizer>(tokenizer_path);
}

void Qwen::head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  bmrt_tensor_with_device(
      &in_tensors[0], logits_mem,
      net->input_dtypes[0], net->stages[0].input_shapes[0]);

  for (int i = 1; i < net->input_num; i++) {
    bmrt_tensor_with_device(
        &in_tensors[i], net->stages[0].input_mems[i],
        net->input_dtypes[i], net->stages[0].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(
        &out_tensors[i], net->stages[0].output_mems[i],
        net->output_dtypes[i], net->stages[0].output_shapes[i]);
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

int Qwen::penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
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
            visited_tokens.begin() + token_length,
            generated_tokens.begin());
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
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());
  
  token_length = tokens.size();

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
  else {
    std::cerr << "\nError: Invalid generation mode.\n";
    std::cerr << "Supported modes are 'greedy' or 'penalty_sample'.\n";
    throw std::runtime_error("Invalid generation mode");
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
    if (io_alone) {
      if (idx == 0) {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      } else {
        d2d(in1_mem, net_blocks_cache[0]->stages[0].input_mems[1]);
        d2d(in2_mem, net_blocks_cache[0]->stages[0].input_mems[2]);
      }
    } else {
      if (idx == 0) {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
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
  else {
    std::cerr << "\nError: Invalid generation mode.\n";
    std::cerr << "Supported modes are 'greedy' or 'penalty_sample'.\n";
    throw std::runtime_error("Invalid generation mode");
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
  if ((int)history_tokens.size() > SEQLEN - 10) {
    history_tokens.clear();
    printf("Error: your question is too large!\n");
    return {};
  }

  std::vector<int> result_tokens;
  int token = forward_first(history_tokens);
  while (token != EOS && token_length < SEQLEN) {
    result_tokens.emplace_back(token);
    token = forward_next();
  }

  return result_tokens;
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
  int tok_num = 0;
  history.emplace_back(std::move(input_str));
  auto input_ids = tk->encode_history(history, SEQLEN);
  token_count = input_ids.size();
  auto time_1 = std::chrono::system_clock::now();
  int pre_token = 0;
  int token = forward_first(input_ids);
  auto time_2 = std::chrono::system_clock::now();
  std::string result;
  while (token != tk->im_end_id && token_count < SEQLEN) {
    std::vector<int> pre_ids = {pre_token};
    std::vector<int> ids = {pre_token, token};
    auto pre_word = tk->decode(pre_ids);
    auto word = tk->decode(ids);
    std::string diff = word.substr(pre_word.size());
    result += diff;
    std::cout << diff << std::flush;
    if (token_count < SEQLEN) {
      token_count++;
    }
    tok_num++;
    token = forward_next();
  }
  auto time_3 = std::chrono::system_clock::now();
  auto ftl_dur =
      std::chrono::duration_cast<std::chrono::microseconds>(time_2 - time_1);
  auto tps_dur =
      std::chrono::duration_cast<std::chrono::microseconds>(time_3 - time_2);
  double tps = tok_num / (tps_dur.count() * 1e-6);
  if (token_count >= SEQLEN) {
    printf(" ......\nWarning: cleanup early history\n");
  }
  // double tht = tokens.size() / (tht_dur.count() * 1e-6);
  printf("\nFTL:%f s, TPS: %f tokens/s\n", ftl_dur.count() * 1e-6, tps);
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
         "  --help                  : Show help info.\n"
         "  --model                 : Set model path \n"
         "  --tokenizer             : Set tokenizer path \n"
         "  --devid                 : Set devices to run for model, e.g. 1,2, if not provided, use 0\n"
         "  --temperature           : Set temperature for generating new token, e.g. 1.0, if not provided, default to 1.0 \n"
         "  --top_p                 : Set top_p for generating new tokens, e.g. 0.8, if not provided, default to 1 \n"
         "  --repeat_penalty        : Set repeat_penalty for generating new tokens, e.g. 1.1, if not provided, default to 1.1 \n"
         "  --repeat_last_n         : Set repeat_penalty for penalizing recent n tokens, e.g. 32, if not provided, default to 32 \n"
         "  --max_new_tokens        : Set max new tokens, e.g. 100, if not provided, stop at EOS or exceeding max length \n"
         "  --generation_mode       : Set generation mode, e.g sample in greedy or penalty_sample, if not provided, default to greedy search \n"
         "  --input_mode            : Set input mode, e.g. unprompted, if not provided, use prompted \n"
         "\n");
}

void processArguments(int argc, char *argv[], std::string &model_path,
                      std::string &tokenizer_path, std::vector<int> &devices,
                      float &temperature, uint16_t &top_p,
                      float &repeat_penalty, int &repeat_last_n,
                      int &max_new_tokens, std::string &generation_mode,
                      std::string &input_mode) {
  struct option longOptions[] = {
      {"model", required_argument, nullptr, 'm'},
      {"tokenizer", required_argument, nullptr, 't'},
      {"devid", required_argument, nullptr, 'd'},
      {"help", no_argument, nullptr, 'h'},
      {"temperature", required_argument, nullptr, 'e'},
      {"top_p", required_argument, nullptr, 'p'},
      {"repeat_penalty", required_argument, nullptr, 'r'},
      {"repeat_last_n", required_argument, nullptr, 'l'},
      {"max_new_tokens", required_argument, nullptr, 'n'},
      {"generation_mode", required_argument, nullptr, 'g'},
      {"input_mode", required_argument, nullptr, 'i'},
      {nullptr, 0, nullptr, 0}};

  int optionIndex = 0;
  int option;

  while ((option = getopt_long(argc, argv, "m:t:d:h:e:p:r:l:n:g", longOptions,
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
    case 'e':
      temperature = std::stof(optarg);
      break;
    case 'p':
      top_p = std::stof(optarg);
      break;
    case 'r':
      repeat_penalty = std::stof(optarg);
      break;
    case 'l':
      repeat_last_n = std::stoi(optarg);
      break;
    case 'n':
      max_new_tokens = std::stoi(optarg);
      break;
    case 'g':
      generation_mode = optarg;
      break;
    case 'i':
      input_mode = optarg;
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
  std::vector<int> devices = {1};
  float temperature = 1.f;
  uint16_t top_p = 1;
  float repeat_penalty = 1.1f;
  int repeat_last_n = 32;
  int max_new_tokens = std::numeric_limits<int>::max();
  std::string generation_mode = "greedy";
  std::string input_mode = "prompted";
  processArguments(argc, argv, model_path, tokenizer_path, devices, temperature,
                   top_p, repeat_penalty, repeat_last_n, max_new_tokens,
                   generation_mode, input_mode);
  if (model_path.empty()) {
    Usage();
    exit(EXIT_FAILURE);
  }

  Qwen qwen;
  printf("Init Environment ...\n");
  qwen.init(devices, model_path, tokenizer_path, temperature, top_p,
            repeat_penalty, repeat_last_n, max_new_tokens, generation_mode,
            input_mode);
  printf("==========================\n");
  qwen.chat();
  qwen.deinit();
  return 0;
}
