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
#include "sentencepiece/sentencepiece_processor.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <map>
#include <random>
#include <vector>

void dump_tensor(bm_handle_t bm_handle, bm_tensor_t &tensor) {
  auto shape = tensor.shape;
  int size = 1;
  for (int i = 0; i < shape.num_dims; ++i) {
    size *= shape.dims[i];
  }
  std::vector<uint16_t> data(size);
  bm_memcpy_d2s(bm_handle, data.data(), tensor.device_mem);
  std::cout << data[0] << "\t" << data[data.size() - 1] << std::endl;
  auto ptr = data.data();
  ptr[0] = ptr[0];
}

static const uint16_t ATTENTION_MASK = 0xF0E2;

typedef union {
  float fval;
  uint32_t bits;
  struct {
    uint32_t frac : 23; // mantissa
    uint32_t exp : 8;   // exponent
    uint32_t sign : 1;  // sign
  } format;
} fp32;

typedef union {
  float fval;
  uint16_t bits;
  struct {
    uint16_t frac : 10; // mantissa
    uint16_t exp : 5;   // exponent
    uint16_t sign : 1;  // sign
  } format;
} fp16;

static inline uint16_t fp32_to_fp16_bits(uint32_t f) {
  /*
   * Extract the sign of the input number into the high bit of the 16-bit word:
   *
   *      +---+-----+-------------+
   *      | S | EEEE E | MMM MMMM |
   *      +---+-----+-------------+
   * Bits  15  14-10      9-0
   */
  const uint32_t sign = f & UINT32_C(0x80000000);
  /*
   * Extract the exponent and the top 7 bits of the mantissa into the bits 0-14
   * of the 16-bit word:
   *
   *      +---+-----+-------------+
   *      | 0 | EEEE E | MMM MMMM |
   *      +---+-----+-------------+
   * Bits  15  14-10      9-0
   */
  const uint32_t rest = (f >> 13) & UINT32_C(0x7FFF);

  // Combine the sign with the rest of the number
  const uint16_t fp16_val = (sign >> 16) | rest;

  // Handle rounding by examining the bits that are being truncated
  const uint32_t rounding_mask = UINT32_C(0x00001FFF);
  const uint32_t rounding_bits = f & rounding_mask;
  const uint32_t halfway = UINT32_C(0x00001000);
  if (rounding_bits > halfway || (rounding_bits == halfway && (fp16_val & 1))) {
    // Round up
    return fp16_val + 1;
  } else {
    // Truncate
    return fp16_val;
  }
}

static inline uint16_t fp32_ieee_to_fp16_value(float val) {
  fp32 f0;
  f0.fval = val;
  return fp32_to_fp16_bits(f0.bits);
}

class LLama2 {
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
  LLama2() : sgen(std::random_device()()) {};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  void load_sentencepiece(std::string tokenizer_path);
  std::string
  build_prompt(std::string query,
               std::vector<std::pair<std::string, std::string>> history);

public:
  int token_length;
  int SEQLEN;     // read from bmodel
  int NUM_LAYERS; // read from bmodel
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
  sentencepiece::SentencePieceProcessor sentencepiece;
  std::vector<std::pair<std::string, std::string>> history_vector;
  std::string sys_config =
      R"(<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n)";
  int EOS;
};

void LLama2::load_sentencepiece(std::string tokenizer_path) {
  printf("Load %s ... ", tokenizer_path.c_str());
  auto status = sentencepiece.Load(tokenizer_path);
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    exit(-1);
  }
  EOS = sentencepiece.eos_id();
  printf("Done!\n");
}

void LLama2::net_launch(const bm_net_info_t *net, int stage_idx) {
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

void LLama2::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void LLama2::head_launch(const bm_net_info_t *net,
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

int LLama2::greedy_search(const bm_net_info_t *net,
                          bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];
  head_launch(net, logits_mem);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

int LLama2::penalty_sample(const bm_net_info_t *net,
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

void LLama2::init(const std::vector<int> &devices, std::string model_path,
                  std::string tokenizer_path, const float &__temperature,
                  const float &__top_p, const float &__repeat_penalty,
                  const int &__repeat_last_n, const int &__max_new_tokens,
                  const std::string &__generation_mode,
                  const std::string &__input_mode) {
  // load tokenizer
  load_sentencepiece(tokenizer_path);

  // generation params
  temperature = __temperature;
  top_p = fp32_to_fp16_bits(__top_p);
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
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
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
  if (net_greedy_head) {
    net_penalty_sample_head =
        bmrt_get_network_info(p_bmrt, "penalty_sample_head");
    NUM_LAYERS = (bmrt_get_network_number(p_bmrt) - 5) / 2;
  } else {
    net_penalty_sample_head = nullptr;
    NUM_LAYERS = (bmrt_get_network_number(p_bmrt) - 3) / 2;
  }
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
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
  }
}

void LLama2::deinit() {
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

int LLama2::forward_first(std::vector<int> &tokens) {
  // make inputs
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());

  token_length = tokens.size();

  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }

  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < token_length; j++) {
      if (j <= i) {
        attention_mask[i * SEQLEN + j] = 0;
      }
    }
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
  if (!net_greedy_head) {
    bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  } else if (generation_mode == "greedy") {
    token = greedy_search(net_greedy_head, lm_out_mem);
  } else if (generation_mode == "penalty_sample") {
    token = penalty_sample(net_penalty_sample_head, lm_out_mem);
  } else {
    std::cerr << "\nError: Invalid generation mode.\n";
    std::cerr << "Supported modes are 'greedy' or 'penalty_sample'.\n";
    throw std::runtime_error("Invalid generation mode");
  }

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

std::string LLama2::build_prompt(
    std::string query,
    std::vector<std::pair<std::string, std::string>> history_vector) {
  std::string prompt = sys_config;
  for (const auto &item : history_vector) {
    prompt += item.first + " [/INST] " + item.second + "</s><s>[INST]] ";
  }
  prompt += query + " [/INST] ";
  return prompt;
}

void LLama2::chat() {
  while (true) {
    std::cout << "\nQuestion: ";
    std::string input_str;
    std::getline(std::cin, input_str);
    if (input_str == "exit") {
      break;
    }

    std::cout << "\nAnswer: " << std::flush;
    answer(input_str);
    std::cout << std::endl;
  }
}

int LLama2::forward_next() {
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
  if (!net_greedy_head) {
    bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  } else if (generation_mode == "greedy") {
    token = greedy_search(net_greedy_head, lm_out_mem);
  } else if (generation_mode == "penalty_sample") {
    token = penalty_sample(net_penalty_sample_head, lm_out_mem);
  } else {
    std::cerr << "\nError: Invalid generation mode.\n";
    std::cerr << "Supported modes are 'greedy' or 'penalty_sample'.\n";
    throw std::runtime_error("Invalid generation mode");
  }

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

void LLama2::answer(const std::string &input_str) {
  std::string sentence_input = build_prompt(input_str, history_vector);

  int tok_num = 0;
  std::vector<int> tokens;
  sentencepiece.Encode(sentence_input, &tokens);

  if (int(tokens.size()) >= SEQLEN - 10) {
    std::cout << "The tokens you input exceeds MAX SEQ LENGTH" << std::endl;
    return;
  }
  int pre_token = 0;
  auto t0 = std::chrono::system_clock::now();
  int token = forward_first(tokens);
  auto t1 = std::chrono::system_clock::now();
  std::string result;
  while (token != EOS && token_length < SEQLEN) {
    std::string pre_word;
    std::string word;
    std::vector<int> pre_ids = {pre_token};
    std::vector<int> ids = {pre_token, token};
    sentencepiece.Decode(pre_ids, &pre_word);
    sentencepiece.Decode(ids, &word);
    std::string diff = word.substr(pre_word.size());
    result += diff;
    std::cout << diff << std::flush;
    tok_num++;
    token = forward_next();
  }
  auto t2 = std::chrono::system_clock::now();
  auto use0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
  auto use1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  printf("\n\nfirst token latency: %f s", (use0.count() * 1e-6));
  printf("\nspeed: %f token/s\n", tok_num / (use1.count() * 1e-6));
  if (token_length >= SEQLEN) {
    history_vector.push_back({input_str, result});
    result.clear();

    size_t half_size = history_vector.size() / 2;
    history_vector.erase(history_vector.begin(),
                         history_vector.begin() + half_size);
  } else {
    history_vector.push_back({input_str, result});
    result.clear();
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
  printf(
      "Usage:\n"
      "  --help                  : Show help info.\n"
      "  --model                 : Set model path \n"
      "  --tokenizer             : Set tokenizer path \n"
      "  --devid                 : Set devices to run for model, e.g. 1,2, if "
      "not provided, use 0\n"
      "  --temperature           : Set temperature for generating new token, "
      "e.g. 1.0, if not provided, default to 1.0 \n"
      "  --top_p                 : Set top_p for generating new tokens, e.g. "
      "0.8, if not provided, default to 1 \n"
      "  --repeat_penalty        : Set repeat_penalty for generating new "
      "tokens, e.g. 1.1, if not provided, default to 1.1 \n"
      "  --repeat_last_n         : Set repeat_penalty for penalizing recent n "
      "tokens, e.g. 32, if not provided, default to 32 \n"
      "  --max_new_tokens        : Set max new tokens, e.g. 100, if not "
      "provided, stop at EOS or exceeding max length \n"
      "  --generation_mode       : Set generation mode, e.g sample in greedy "
      "or penalty_sample, if not provided, default to greedy search \n"
      "  --input_mode            : Set input mode, e.g. unprompted, if not "
      "provided, use prompted \n"
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
  printf("Demo for Llama\n");
  std::string model_path;
  std::string tokenizer_path;
  std::vector<int> devices = {0};
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

  LLama2 llama2;
  printf("Init Environment ...\n");
  llama2.init(devices, model_path, tokenizer_path, temperature, top_p,
              repeat_penalty, repeat_last_n, max_new_tokens, generation_mode,
              input_mode);
  printf("==========================\n");
  llama2.chat();
  llama2.deinit();
  return 0;
}
