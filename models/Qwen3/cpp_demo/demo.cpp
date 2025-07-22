//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "bmruntime_interface.h"
#include "json.hpp"
#include "tokenizers-cpp/tokenizers_cpp.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory.h>
#include <random>
#include <vector>

using tokenizers::Tokenizer;

static inline std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open [ " << path << " ]" << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

bool ends_with(const std::string &str, const std::string &suffix) {
  if (str.size() < suffix.size())
    return false;
  return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

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

struct GenerationConfig {
  std::vector<int> eos_token_id;
  float repetition_penalty = 1.0;
  float temperature = 1.0;
  int top_k = 50;
  float top_p = 1.0;
  std::vector<std::string> stop_strings;
  static GenerationConfig from_json(const std::string &path) {
    GenerationConfig config;
    std::ifstream in(path);
    nlohmann::json j;
    in >> j;
    if (j.contains("eos_token_id"))
      config.eos_token_id = j["eos_token_id"].get<std::vector<int>>();
    if (j.contains("repetition_penalty"))
      config.repetition_penalty = j["repetition_penalty"].get<float>();
    if (j.contains("temperature"))
      config.temperature = j["temperature"].get<float>();
    if (j.contains("top_k"))
      config.top_k = j["top_k"].get<int>();
    if (j.contains("top_p"))
      config.top_p = j["top_p"].get<float>();
    if (j.contains("stop_strings"))
      config.stop_strings = j["stop_strings"].get<std::vector<std::string>>();
    return config;
  }
};

class Qwen3 {
public:
  void init(std::string model_path, std::string config_path,
            std::string system_prompt, bool enable_history, bool do_sample,
            const std::vector<int> &devid);
  void deinit();
  void chat();
  void answer(const std::string input_str);
  int forward_first(std::vector<int> &tokens);
  int forward_next();
  void clear_kv();
  std::string build_prompt(std::string input_str);
  std::mt19937 sgen;
  Qwen3() : sgen(std::random_device()()) {};

private:
  int forward_first_with_kv(std::vector<int> &tokens);
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  void net_launch_dyn(const bm_net_info_t *net, int real_len,
                      int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, size_t offset = 0,
                  size_t size = 0);
  int greedy_search(bm_device_mem_t &logits_mem);
  int penalty_sample(bm_device_mem_t &logits_mem);
  void init_by_names();

public:
  int token_length;
  int SEQLEN;
  int MAX_INPUT_LENGTH;
  int PREFILL_KV_LENGTH;
  int NUM_LAYERS;
  int hidden_bytes;
  int kv_bytes;
  bool is_dynamic;
  bool enable_history;
  bool support_prefill_kv;
  int history_length;
  uint16_t mask_value;
  std::vector<int> visited_tokens;
  std::vector<std::pair<std::string, std::string>> history_vector;
  std::string sys_config;
  // generation
  std::string generation_mode;
  std::vector<std::string> stop_strings;
  float penalty;
  float temperature;
  int top_k;
  float top_p;

private:
  bm_handle_t bm_handle;
  std::vector<bm_handle_t> handles;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_vit;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm_head;
  const bm_net_info_t *net_greedy_head;
  const bm_net_info_t *net_sample_head;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
  // tokenizer & processor
  std::unique_ptr<Tokenizer> tok;
  std::vector<int> EOS;
};

void Qwen3::net_launch(const bm_net_info_t *net, int stage_idx) {
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

void Qwen3::net_launch_dyn(const bm_net_info_t *net, int real_len,
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

void Qwen3::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, size_t offset,
                size_t size) {
  if (!size)
    size = bm_mem_get_device_size(src);
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void Qwen3::init_by_names() {
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
  net_lm_head = bmrt_get_network_info(p_bmrt, "lm_head");
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
    enable_history = true;
    PREFILL_KV_LENGTH = net_blocks[0]->stages[0].input_shapes[3].dims[1];
    printf("History by kv: True\n");
  }
}

void Qwen3::init(std::string model_path, std::string config_path,
                 std::string system_prompt, bool save_history, bool do_sample,
                 const std::vector<int> &devices) {
  sys_config = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
  enable_history = save_history;

  // load tokenizer
  std::string tokenizer_path = config_path + "/tokenizer.json";
  std::cout << "Processor [" << tokenizer_path.c_str() << "] loading .... ";
  auto blob = LoadBytesFromFile((tokenizer_path).c_str());
  tok = Tokenizer::FromBlobJSON(blob);
  EOS.push_back(tok->TokenToId("<|endoftext|>"));
  EOS.push_back(tok->TokenToId("<|im_end|>"));
  std::cout << "Done!" << std::endl;

  // load generation config
  generation_mode = "greedy";
  if (do_sample) {
    generation_mode = "sample";
    std::string generation_path = config_path + "/generation_config.json";
    std::cout << "Generation Config [" << generation_path.c_str()
              << "] loading .... ";
    auto gen_config = GenerationConfig::from_json(generation_path);
    penalty = gen_config.repetition_penalty;
    temperature = gen_config.temperature;
    top_k = gen_config.top_k;
    top_p = gen_config.top_p;
    for (auto id : gen_config.eos_token_id) {
      EOS.push_back(id);
    }
    if (!gen_config.stop_strings.empty()) {
      stop_strings = gen_config.stop_strings;
    }
    std::cout << "Done!" << std::endl;
  }

  // request bm_handle
  std::cout << "Device [ ";
  for (auto d : devices) {
    std::cout << d << " ";
  }
  std::cout << "] loading .... ";
  for (auto d : devices) {
    bm_handle_t h;
    bm_status_t status = bm_dev_request(&h, d);
    assert(BM_SUCCESS == status);
    handles.push_back(h);
  }
  bm_handle = handles[0];
  std::cout << "Done!" << std::endl;

  p_bmrt = bmrt_create_ex(handles.data(), handles.size());
  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  // load bmodel
  std::cout << "Qwen3 [" << model_path.c_str() << "] loading .... ";
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  std::cout << "Done!" << std::endl;

  // init networks
  init_by_names();

  hidden_bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[0]);
  kv_bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);

  auto buffer_size =
      bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);

  bm_set_device_mem(&net_embed->stages[0].output_mems[0], dev_buffer.size,
                    dev_buffer.u.device.device_addr);

  visited_tokens.resize(SEQLEN);
  if (net_embed_cache->output_dtypes[0] == BM_FLOAT16) {
    mask_value = 0xF0E2; // ATTENTION_MASK in fp16
  } else if (net_embed_cache->output_dtypes[0] == BM_BFLOAT16) {
    mask_value = 0xC61C; // ATTENTION_MASK in bfloat16
  } else {
    std::cerr << "\nError: Unsupported Dtype: "
              << net_embed_cache->output_dtypes[0];
    throw std::runtime_error("Invalid attention dtype");
  }

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

void Qwen3::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

int Qwen3::greedy_search(bm_device_mem_t &logits_mem) {
  auto &out_mem = net_greedy_head->stages[0].output_mems[0];
  bm_set_device_mem(&net_greedy_head->stages[0].input_mems[0], logits_mem.size,
                    logits_mem.u.device.device_addr);
  net_launch(net_greedy_head);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

int Qwen3::penalty_sample(bm_device_mem_t &logits_mem) {
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

int Qwen3::forward_first(std::vector<int> &inputs) {
  if (support_prefill_kv) {
    return forward_first_with_kv(inputs);
  }
  std::vector<int> position_id(MAX_INPUT_LENGTH, 0);
  std::copy(inputs.begin(), inputs.end(), visited_tokens.data());
  token_length = inputs.size();
  std::vector<uint16_t> attention_mask(MAX_INPUT_LENGTH * MAX_INPUT_LENGTH,
                                       mask_value);
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
  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  // forward embeding
  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)visited_tokens.data());
  net_launch(net_embed);

  // forward blocks
  empty_net(bm_handle, net_blocks[0]);
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    d2d(in0_mem, out_mem, 0, token_length * hidden_bytes);
    if (idx == 0) {
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
  auto &lm_in_mem = net_lm_head->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm_head->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (token_length - 1) * hidden_bytes, hidden_bytes);
  net_launch(net_lm_head);
  int token = 0;
  if (!net_greedy_head && !net_sample_head) {
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

int Qwen3::forward_first_with_kv(std::vector<int> &inputs) {
  int max_kv_length = MAX_INPUT_LENGTH + PREFILL_KV_LENGTH;
  std::vector<int> position_id(MAX_INPUT_LENGTH, 0);
  std::copy(inputs.begin(), inputs.end(), visited_tokens.data());
  auto old_length = history_length;
  token_length = inputs.size();
  history_length += token_length;
  std::vector<uint16_t> attention_mask(MAX_INPUT_LENGTH * max_kv_length,
                                       mask_value);
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
  bm_memcpy_s2d(bm_handle, in_mem, (void *)inputs.data());
  net_launch(net_embed);

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
  auto &lm_in_mem = net_lm_head->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm_head->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (token_length - 1) * hidden_bytes, hidden_bytes);
  net_launch(net_lm_head);
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

int Qwen3::forward_next() {
  int cur_token = visited_tokens[token_length - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = history_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = mask_value;
  }
  int32_t position_id = history_length - 1;
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
  auto &lm_in_mem = net_lm_head->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm_head->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm_head);
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

void Qwen3::clear_kv() {
  if (!support_prefill_kv) {
    return;
  }
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  history_length = 0;
}

std::string Qwen3::build_prompt(std::string input_str) {
  std::string prompt;
  if (history_length == 0) {
    prompt = sys_config;
  }
  prompt += "<|im_start|>user\n";
  if (enable_history && !support_prefill_kv) {
    for (const auto &item : history_vector) {
      prompt += item.first + "<|im_end|>\n" + "<|im_start|>assistant\n" +
                item.second + "<|im_end|>\n<|im_start|>user\n";
    }
  }
  prompt += input_str + "<|im_end|>\n<|im_start|>assistant\n";
  return prompt;
}

void Qwen3::answer(const std::string input_str) {
  std::string sentence_input = build_prompt(input_str);
  std::vector<int> tokens = tok->Encode(sentence_input);
  int pre_token = 0;
  int tok_num = 0;
  if ((int)tokens.size() > MAX_INPUT_LENGTH) {
    std::cerr << "Error: Input length exceeds maximum input length of "
              << MAX_INPUT_LENGTH << std::endl;
    return;
  }
  auto t0 = std::chrono::system_clock::now();
  int token = forward_first(tokens);
  auto t1 = std::chrono::system_clock::now();

  std::string result;
  while (std::find(EOS.begin(), EOS.end(), token) == EOS.end() &&
         token_length < SEQLEN) {
    std::vector<int> pre_ids = {pre_token};
    std::vector<int> ids = {pre_token, token};
    std::string pre_word = tok->Decode(pre_ids);
    std::string word = tok->Decode(ids);
    std::string diff = word.substr(pre_word.size());
    result += diff;
    bool stop_by_string = false;
    for (const auto &stop : stop_strings) {
      if (ends_with(result, stop)) {
        stop_by_string = true;
        break;
      }
    }
    if (stop_by_string)
      break;
    std::cout << diff << std::flush;
    tok_num++;
    token = forward_next();
  }
  auto t2 = std::chrono::system_clock::now();
  auto use0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
  auto use1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << std::endl;
  std::cout << "FTL: " << (use0.count() * 1e-6) << " s" << std::endl;
  std::cout << "TPS: " << tok_num / (use1.count() * 1e-6) << " token/s"
            << std::endl;
  if (support_prefill_kv) {
    if (history_length >= SEQLEN) {
      clear_kv();
    }
  } else if (enable_history) {
    if (token_length >= SEQLEN) {
      history_vector.push_back({input_str, result});
      size_t half_size = history_vector.size() / 2;
      history_vector.erase(history_vector.begin(),
                           history_vector.begin() + half_size);
      std::cout << "history length exceed max sequence length, erase half"
                << std::endl;
    } else {
      history_vector.push_back({input_str, result});
    }
  }
  result.clear();
}

void Qwen3::chat() {
  std::cout
      << "================================================================="
      << std::endl
      << "1. If you want to quit, please enter one of [q, quit, exit]"
      << std::endl
      << "2. To create a new chat session, please enter one of [clear, new]"
      << std::endl
      << "================================================================="
      << std::endl;
  while (true) {
    std::cout << "\nQuestion: ";
    std::string input_str;
    std::getline(std::cin, input_str);
    if (input_str == "exit" || input_str == "q" || input_str == "quit") {
      break;
    }
    if (input_str == "clear" || input_str == "new") {
      history_vector = {};
      clear_kv();
      std::cout << "New chat session created." << std::endl;
      continue;
    }
    std::cout << "\nAnswer: " << std::flush;
    answer(input_str);
    std::cout << std::endl;
  }
}

void Usage() {
  printf("Usage:\n"
         "  -h, --help      : Show help info.\n"
         "  -m, --model     : Set model path \n"
         "  -c, --config    : Set processor config path \n"
         "  -e, --enable_history : if set, enable history memory\n"
         "  -s, --do_sample : if set, sample by generation config\n"
         "  -d, --devid     : Set devices to run for model, default is '0'\n");
}

void processArguments(int argc, char *argv[], std::string &model_path,
                      std::string &config_path, std::vector<int> &devices,
                      bool &enable_history, bool &do_sample) {
  struct option longOptions[] = {{"model", required_argument, nullptr, 'm'},
                                 {"config", required_argument, nullptr, 'c'},
                                 {"devid", required_argument, nullptr, 'd'},
                                 {"enable_history", no_argument, nullptr, 'e'},
                                 {"do_sample", no_argument, nullptr, 's'},
                                 {"help", no_argument, nullptr, 'h'},
                                 {nullptr, 0, nullptr, 0}};

  int optionIndex = 0;
  int option;

  while ((option = getopt_long(argc, argv, "m:c:d:esh", longOptions,
                               &optionIndex)) != -1) {
    switch (option) {
    case 'm':
      model_path = optarg;
      break;
    case 'c':
      config_path = optarg;
      break;
    case 'd':
      devices = {atoi(optarg)};
      break;
    case 'e':
      enable_history = true;
      break;
    case 's':
      do_sample = true;
      break;
    case 'h':
    case '?':
      Usage();
      exit(EXIT_SUCCESS);
    default:
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char **argv) {
  std::string model_path;
  std::string config_path = "../../config";
  std::vector<int> devices = {0};
  bool enable_history = false;
  bool do_sample = false;

  processArguments(argc, argv, model_path, config_path, devices, enable_history,
                   do_sample);
  if (model_path.empty()) {
    Usage();
    exit(EXIT_FAILURE);
  }

  std::string system_prompt = "You are a helpful assistant.";

  Qwen3 model;
  std::cout << "Init Environment ..." << std::endl;
  model.init(model_path, config_path, system_prompt, enable_history, do_sample,
             devices);
  model.chat();
  model.deinit();
  return 0;
}
