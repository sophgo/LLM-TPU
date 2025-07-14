//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "bmruntime_interface.h"
#include "cv_utils.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory.h>
#include <random>
#include <tokenizers-cpp/tokenizers_cpp.h>
#include <vector>

using tokenizers::Tokenizer;

static const int VISION_PAD_TOKEN = 151654;
static const int IMAGE_PAD_TOKEN = 151655;

static const uint16_t ATTENTION_MASK = 0xC61C; // -9984 by bfloat16

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

class Qwen2VL {
public:
  void init(std::string model_path, std::string config_path,
            std::string system_prompt, bool enable_history,
            const std::vector<int> &devid);
  void deinit();
  void chat(std::string image_path);
  void answer(const std::string input_str, std::string image_path);
  int forward_first(std::vector<int> &tokens, std::vector<float> &pixel_values);
  int forward_next();
  std::string build_prompt(std::string input_str);
  std::mt19937 sgen;
  Qwen2VL() : sgen(std::random_device()()) {};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, size_t offset = 0,
                  size_t size = 0);

public:
  int token_length;
  int SEQLEN;
  int NUM_LAYERS;
  int MAX_PATCHES;
  int VIT_DIMS;
  int hidden_bytes;
  int kv_bytes;
  bool enable_history;
  uint16_t mask_value;
  std::vector<int> visited_tokens;
  std::vector<std::pair<std::string, std::string>> history_vector;
  std::string sys_config;

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
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;

  // tokenizer & processor
  std::unique_ptr<Tokenizer> tok;
  std::unique_ptr<Maker> maker;
  Config config;
  int EOS;
  int ID_IM_END;
};

void Qwen2VL::net_launch(const bm_net_info_t *net, int stage_idx) {
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

void Qwen2VL::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, size_t offset,
                  size_t size) {
  if (!size)
    size = bm_mem_get_device_size(src);
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void Qwen2VL::init(std::string model_path, std::string config_path,
                   std::string system_prompt, bool save_history,
                   const std::vector<int> &devices) {
  sys_config = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
  enable_history = save_history;

  // load tokenizer
  std::cout << "Config [" << config_path.c_str() << "] loading .... ";
  auto blob = LoadBytesFromFile((config_path + "/tokenizer.json").c_str());
  tok = Tokenizer::FromBlobJSON(blob);
  EOS = tok->TokenToId("<|endoftext|>");
  ID_IM_END = tok->TokenToId("<|im_end|>");
  std::cout << "Done!" << std::endl;

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
  std::cout << "Qwen2VL [" << model_path.c_str() << "] loading .... ";
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  std::cout << "Done!" << std::endl;

  // init networks
  net_vit = bmrt_get_network_info(p_bmrt, "vit");
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_lm_head = bmrt_get_network_info(p_bmrt, "lm_head");
  net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
  if (net_greedy_head) {
    NUM_LAYERS = (bmrt_get_network_number(p_bmrt) - 6) / 2;
  } else {
    NUM_LAYERS = (bmrt_get_network_number(p_bmrt) - 4) / 2;
  }
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }

  // init parameters
  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1];
  MAX_PATCHES = net_vit->stages[0].input_shapes[0].dims[0];
  VIT_DIMS = net_vit->stages[0].input_shapes[0].dims[1];
  hidden_bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[0]);
  kv_bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  visited_tokens.resize(SEQLEN);

  mask_value = ATTENTION_MASK;

  // empty networks
  empty_net(bm_handle, net_vit);
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty_net(bm_handle, net_blocks[i]);
    empty_net(bm_handle, net_blocks_cache[i]);
  }

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
  }

  // load processor
  config.model_type = "qwen2_vl";
  config.temporal_patch_size = 2;
  config.spatial_merge_size = 2;
  config.patch_size = 14;
  config.MAX_PATCHES = MAX_PATCHES;
  config.SEQLEN = SEQLEN;
  maker = std::make_unique<Maker>(config);
}

void Qwen2VL::deinit() {
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

int Qwen2VL::forward_first(std::vector<int> &raw_tokens,
                           std::vector<float> &pixel_values) {
  std::vector<int> tokens = maker->insert_tokens(raw_tokens, IMAGE_PAD_TOKEN);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());
  token_length = tokens.size();
  std::vector<int> media_offset;
  std::vector<int> media_size;
  get_media_info(tokens, media_offset, media_size, IMAGE_PAD_TOKEN);
  config.media_offset = media_offset[0];
  config.media_size = media_size[0];
  config.total_length = token_length;

  std::vector<float> pixel_values_pad(MAX_PATCHES * VIT_DIMS, 0);
  std::copy(pixel_values.begin(), pixel_values.end(), pixel_values_pad.data());
  auto vit_position_id = maker->make_vit_position_id();
  auto vit_attention_mask = maker->make_vit_attention_mask();
  auto position_id = maker->make_position_id();
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, mask_value);
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j <= i; j++) {
      attention_mask[i * SEQLEN + j] = 0;
    }
  }

  // forward embeding
  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)visited_tokens.data());
  net_launch(net_embed);

  // forward vit
  auto &vit_in0_mem = net_vit->stages[0].input_mems[0];
  auto &vit_in1_mem = net_vit->stages[0].input_mems[1];
  auto &vit_in2_mem = net_vit->stages[0].input_mems[2];
  auto &vit_out_mem = net_vit->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, vit_in0_mem, (void *)pixel_values_pad.data());
  bm_memcpy_s2d(bm_handle, vit_in1_mem, (void *)vit_position_id.data());
  bm_memcpy_s2d(bm_handle, vit_in2_mem, (void *)vit_attention_mask.data());
  net_launch(net_vit);
  bm_memcpy_d2d_byte(bm_handle, out_mem, config.media_offset * hidden_bytes,
                     vit_out_mem, 0, config.media_size * hidden_bytes);

  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    d2d(in0_mem, out_mem, 0, token_length * hidden_bytes);
    if (idx == 0) {
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    }
    net_launch(net_blocks[idx]);
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
  if (net_greedy_head) {
    auto &head_in_mem = net_greedy_head->stages[0].input_mems[0];
    auto &head_out_mem = net_greedy_head->stages[0].output_mems[0];
    d2d(head_in_mem, lm_out_mem);
    net_launch(net_greedy_head);
    bm_memcpy_d2s(bm_handle, (void *)&token, head_out_mem);
  } else {
    bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  }
  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

int Qwen2VL::forward_next() {
  int cur_token = visited_tokens[token_length - 1];
  config.total_length = token_length;

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = mask_value;
  }
  auto position_id = maker->make_next_position_id();

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
    d2d(in0_mem, out_mem, 0, hidden_bytes);
    if (idx == 0) {
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
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

  // lmhead
  auto &lm_in_mem = net_lm_head->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm_head->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm_head);
  int token = 0;
  if (net_greedy_head) {
    auto &head_in_mem = net_greedy_head->stages[0].input_mems[0];
    auto &head_out_mem = net_greedy_head->stages[0].output_mems[0];
    d2d(head_in_mem, lm_out_mem);
    net_launch(net_greedy_head);
    bm_memcpy_d2s(bm_handle, (void *)&token, head_out_mem);
  } else {
    bm_memcpy_d2s(bm_handle, (void *)&token, lm_out_mem);
  }
  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

std::string Qwen2VL::build_prompt(std::string input_str) {
  std::string prompt = sys_config;
  prompt += "<|im_start|>user\n<|vision_start|><|vision_pad|><|vision_end|>";
  if (enable_history) {
    for (const auto &item : history_vector) {
      prompt += item.first + "<|im_end|>\n" + "<|im_start|>assistant\n" +
                item.second + "<|im_end|>\n<|im_start|>user\n";
    }
  }
  prompt += input_str + "<|im_end|>\n<|im_start|>assistant\n";
  return prompt;
}

void Qwen2VL::answer(const std::string input_str,
                     const std::string image_path) {
  std::string sentence_input = build_prompt(input_str);
  std::vector<int> tokens = tok->Encode(sentence_input);
  std::replace(tokens.begin(), tokens.end(), VISION_PAD_TOKEN, IMAGE_PAD_TOKEN);

  std::vector<float> pixel_values = process_image(image_path, config);

  int pre_token = 0;
  int tok_num = 0;

  auto t0 = std::chrono::system_clock::now();
  int token = forward_first(tokens, pixel_values);
  auto t1 = std::chrono::system_clock::now();

  std::string result;
  while (token != EOS && token != ID_IM_END && token_length < SEQLEN) {
    std::vector<int> pre_ids = {pre_token};
    std::vector<int> ids = {pre_token, token};
    std::string pre_word = tok->Decode(pre_ids);
    std::string word = tok->Decode(ids);
    std::string diff = word.substr(pre_word.size());
    result += diff;
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

  if (enable_history) {
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

void Qwen2VL::chat(std::string image_path) {
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
      std::cout << "\nNew Image Path: ";
      std::getline(std::cin, image_path);
      continue;
    }
    std::cout << "\nAnswer: " << std::flush;
    answer(input_str, image_path);
    std::cout << std::endl;
  }
}

void Usage() {
  printf("Usage:\n"
         "  -h, --help      : Show help info.\n"
         "  -m, --model     : Set model path \n"
         "  -c, --config    : Set config path, default is '../config'\n"
         "  -i, --image     : Set image path, default is '../test.jpg' \n"
         "  -d, --devid     : Set devices to run for model, default is '0'\n"
         "  -e, --enable_history : if set, enable history memory\n");
}

void processArguments(int argc, char *argv[], std::string &model_path,
                      std::string &config_path, std::string &image_path,
                      std::vector<int> &devices, bool &enable_history) {
  struct option longOptions[] = {{"model", required_argument, nullptr, 'm'},
                                 {"config", required_argument, nullptr, 'c'},
                                 {"image", required_argument, nullptr, 'i'},
                                 {"devid", required_argument, nullptr, 'd'},
                                 {"enable_history", no_argument, nullptr, 'e'},
                                 {"help", no_argument, nullptr, 'h'},
                                 {nullptr, 0, nullptr, 0}};

  int optionIndex = 0;
  int option;

  while ((option = getopt_long(argc, argv, "m:c:i:d:eh:", longOptions,
                               &optionIndex)) != -1) {
    switch (option) {
    case 'm':
      model_path = optarg;
      break;
    case 'c':
      config_path = optarg;
      break;
    case 'i':
      image_path = optarg;
      break;
    case 'd':
      devices = {atoi(optarg)};
      break;
    case 'e':
      enable_history = true;
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
  std::string model_path;
  std::string config_path;
  std::string image_path;
  std::vector<int> devices = {0};
  bool enable_history = false;

  processArguments(argc, argv, model_path, config_path, image_path, devices,
                   enable_history);
  if (model_path.empty()) {
    Usage();
    exit(EXIT_FAILURE);
  }
  if (config_path.empty()) {
    config_path = "../processor";
  }
  if (image_path.empty()) {
    image_path = "../test.jpg";
  }

  std::string system_prompt = "You are a helpful assistant.";

  Qwen2VL model;
  std::cout << "Init Environment ..." << std::endl;
  model.init(model_path, config_path, system_prompt, enable_history, devices);
  model.chat(image_path);
  model.deinit();
  return 0;
}
