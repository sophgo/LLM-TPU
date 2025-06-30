//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <getopt.h>
#include <inttypes.h>
#include <iostream>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdio.h>
#include <vector>

#include "bmruntime_interface.h"
#include "memory.h"
#include "utils.h"

static const float ATTENTION_MASK = -10000.;
typedef uint8_t *(*decrypt_func)(const uint8_t *, uint64_t, uint64_t *);

class Qwen {
public:
  void load_bmodel(const std::vector<int> &devices,
                   const std::string &model_path);
  void init_nets();
  void init_params();

  void init(const std::vector<int> &devid, const std::string &model_path,
            bool read_bmodel = true);
  void deinit();
  void init_decrypt();
  void deinit_decrypt();
  int forward_first(std::vector<int> &tokens);
  int forward_next();
  void update_bmodel_weight(
    const std::string &model_path, const std::string &lora_path, const std::string &net_idx,
    const std::string &mem_idx, const std::vector<std::string> &weight_idx);
  void empty_bmodel_weight(
    const std::string &model_path, const std::string &net_idx,
    const std::string &mem_idx, const std::vector<std::string> &weight_idx);
  std::vector<int> generate(std::vector<int> &history_tokens, int EOS);

  std::mt19937 sgen;
  Qwen();
  ~Qwen();

private:
  // d2d
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, size_t offset);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, size_t offset,
                  size_t size);

  // infernece
  std::vector<uint16_t>
  load_and_infer_embedding(const std::vector<int> &tokens);
  void net_launch(const bm_net_info_t *net, int stage_idx);
  void dynamic_net_launch(const bm_net_info_t *net, int token_length,
                          int stage_idx);

  bm_device_mem_t embedding_launch(const bm_net_info_t *net0,
                                   const bm_net_info_t *net1,
                                   const bm_net_info_t *net2,
                                   const std::vector<int> &tokens);
  bm_device_mem_t lm_launch(const bm_net_info_t *net,
                            const bm_device_mem_t &out_mem, size_t offset,
                            size_t size);

  // tensors
  void make_in_tensors();
  void free_in_tensors();

  // sample
  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem,
                   int stage_idx);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem,
                     std::vector<int> &input_tokens, int &token_length);

  // error
  void handle_error();
  void bmrt_error();
  void bmodel_error();
  void launch_error();
  void ioalone_error();

public:
  bool is_dynamic;
  uint32_t prefill_reuse;
  std::vector<int> total_tokens;
  std::string lib_path;
  std::string embedding_path;
  int status_code;
  int stage_idx;
  bool enable_lora_embedding;
  bool make_in_tensors_flag;
  uint64_t header_size;

  // model
  int hidden_bytes;
  int kv_bytes;
  int prefill_length;
  int total_length;
  int lora_embedding_flag;
  int SEQLEN;
  int NUM_LAYERS;
  int MAX_PREFILL_LENGTH;
  int BATCH_SIZE;

  // generation
  float temperature;
  float top_p;
  float repeat_penalty;
  int repeat_last_n;
  int max_new_tokens;
  std::string generation_mode;

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lora_embed, *net_lora_embed_cache;
  const bm_net_info_t *net_lm, *net_greedy_head, *net_penalty_sample_head;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
  bm_tensor_t inputs_pid, next_pid;
  bm_tensor_t inputs_attention, next_attention;

  uint16_t mask_value;
  void *decrypt_handle_;      // handle of decrypt lib
  decrypt_func decrypt_func_; // decrypt func from lib
};

// init
Qwen::Qwen() {
  prefill_reuse = 0;
  stage_idx = 0;
  status_code = 0;
  total_tokens.clear();
  enable_lora_embedding = false;
  make_in_tensors_flag = false;
  header_size = 64;

  // path
  lib_path = "";
  embedding_path = "";

  // length
  prefill_length = 0;
  total_length = 0;
  SEQLEN = 0;
  NUM_LAYERS = 0;
  MAX_PREFILL_LENGTH = 0;

  // 
  sgen = std::mt19937(std::random_device()());
  bm_handle = nullptr;
  p_bmrt = nullptr;
  decrypt_handle_ = nullptr;
  decrypt_func_ = nullptr;
}

void Qwen::deinit() {
  deinit_decrypt();

  if (handles.empty()) {
    return;
  }

  if (make_in_tensors_flag) {
    free_in_tensors();
  }

  if (prefill_reuse == 1) {
    // TODO
  }

  if (p_bmrt != nullptr) {
    bmrt_destroy(p_bmrt);
    p_bmrt = nullptr;
  }

  for (auto h : handles) {
    bm_dev_free(h);
  }
  handles.clear();
}

Qwen::~Qwen() {
  deinit();
}

static inline void ASSERT(bool ret) {
  if (!ret) {
    throw std::runtime_error("runtime error");
  }
}

static inline void ASSERT(bool ret, std::string message) {
  if (!ret) {
    throw std::runtime_error(message);
  }
}

void Qwen::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(dst));
}

void Qwen::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, size_t offset) {
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0,
                     bm_mem_get_device_size(src));
}

void Qwen::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, size_t offset,
               size_t size) {
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

//===------------------------------------------------------------===//
// Decrypt
//===------------------------------------------------------------===//
void Qwen::init_decrypt() {
  // init decrypt
  if (lib_path.empty()) {
    return;
  }
  decrypt_handle_ = dlopen(lib_path.c_str(), RTLD_LAZY);
  if (!decrypt_handle_) {
    std::cout << "Error:"
              << "Decrypt lib [" << lib_path << "] load failed." << std::endl;
    throw std::runtime_error("");
  }
  decrypt_func_ = (decrypt_func)dlsym(decrypt_handle_, "decrypt");
  auto error = dlerror();
  if (error) {
    dlclose(decrypt_handle_);
    std::cout << "Error:"
              << "Decrypt lib [" << lib_path << "] symbol find failed."
              << std::endl;
    throw std::runtime_error("");
  }
  return;
}

void Qwen::deinit_decrypt() {
  // Step 1: Close the dynamic library handle if it's open.
  if (!lib_path.empty() && decrypt_handle_) {
    dlclose(decrypt_handle_);
    decrypt_handle_ =
        nullptr; // Avoid dangling pointer by resetting to nullptr.
  }

  // Step 2: Reset the function pointer to nullptr.
  // No need to free or close anything specific for it.
  decrypt_func_ = nullptr;
}

//===------------------------------------------------------------===//
// Exception
//===------------------------------------------------------------===//
// can not create handle
void Qwen::handle_error() {
  status_code = -2;
  throw std::runtime_error("can not create handle");
}

// can not create bmrt
void Qwen::bmrt_error() {
  status_code = -3;
  throw std::runtime_error("can not create bmrt");
}

// can not load bmodel
void Qwen::bmodel_error() {
  status_code = -4;
  throw std::runtime_error("can not load bmodel correctly");
}

// can not inference bmodel
void Qwen::launch_error() {
  status_code = -5;
  throw std::runtime_error("can not inference bmodel");
}

// addr_mode = 0, but must set addr_mode =1
void Qwen::ioalone_error() {
  status_code = -6;
  throw std::runtime_error(
      "addr_mode = 0 in your bmodel, but must set addr_mode = 1");
}

void Qwen::head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem,
                       int stage_idx) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  bmrt_tensor_with_device(&in_tensors[0], logits_mem, net->input_dtypes[0],
                          net->stages[stage_idx].input_shapes[0]);

  for (int i = 1; i < net->input_num; i++) {
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
  if (!ret) {
    launch_error();
  } else {
    bm_thread_sync(bm_handle);
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
  if (!ret) {
    launch_error();
  } else {
    bm_thread_sync(bm_handle);
  }
}

void Qwen::dynamic_net_launch(const bm_net_info_t *net, int token_length,
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
  in_tensors[0].shape.dims[1] = token_length;
  in_tensors[1].shape.dims[1] = token_length;
  in_tensors[2].shape.dims[2] = token_length;
  in_tensors[2].shape.dims[3] = token_length;
  out_tensors[0].shape.dims[1] = token_length;
  out_tensors[1].shape.dims[1] = token_length;
  out_tensors[2].shape.dims[1] = token_length;
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  if (!ret) {
    launch_error();
  } else {
    bm_thread_sync(bm_handle);
  }
}

void Qwen::load_bmodel(const std::vector<int> &devices,
                       const std::string &model_path) {
  // request bm_handle
  std::cout << "Device [ ";
  for (auto d : devices) {
    std::cout << d << " ";
  }
  std::cout << "] loading ....\n";
  for (auto d : devices) {
    bm_handle_t h;
    bm_status_t status = bm_dev_request(&h, d);
    if (BM_SUCCESS != status) {
      handle_error();
    }
    handles.push_back(h);
  }
  bm_handle = handles[0];

  // create bmruntime
#ifdef SOC_TARGET
  p_bmrt = bmrt_create(handles[0]);
#else
  p_bmrt = bmrt_create_ex(handles.data(), handles.size());
#endif
  if (NULL == p_bmrt) {
    bmrt_error();
  }
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

  if (!ret) {
    bmodel_error();
  }
  printf("Done!\n");
}

void Qwen::init_nets() {
  // net embed and lm_head
  ASSERT(bmrt_get_network_index(p_bmrt, "embedding") != -1 ||
         !embedding_path.empty(), "bmodel is lack of embedding or embedding_path is empty");
  if (embedding_path.empty()) {
    net_embed = bmrt_get_network_info(p_bmrt, "embedding");
    net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  }
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
  net_penalty_sample_head =
      bmrt_get_network_info(p_bmrt, "penalty_sample_head");

  lora_embedding_flag = bmrt_get_network_index(p_bmrt, "lora_embedding");
  auto num_nets = bmrt_get_network_number(p_bmrt);
  if (!embedding_path.empty()) {
    num_nets = num_nets - 2;
  }
  if (lora_embedding_flag != -1) {
    net_lora_embed = bmrt_get_network_info(p_bmrt, "lora_embedding");
    net_lora_embed_cache = bmrt_get_network_info(p_bmrt, "lora_embedding_cache");
    num_nets = num_nets - 2;
  }
  NUM_LAYERS = num_nets / 2;

  // net blocks
  net_blocks.clear();
  net_blocks_cache.clear();
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }

  // convert attention to uint16_t
  if (net_blocks[0]->input_dtypes[0] == BM_FLOAT16) {
    mask_value = fp32_to_fp16_bits(ATTENTION_MASK);
  } else if (net_blocks[0]->input_dtypes[0] == BM_BFLOAT16) {
    mask_value = fp32_to_bf16_bits(ATTENTION_MASK);
  } else {
    std::cerr << "\nError: Invalid attention dtype\n";
    std::cerr << "Supported dtype are 'BM_FLOAT16' or 'BM_BFLOAT16'\n";
    throw std::runtime_error("Invalid attention dtype");
  }
}

void Qwen::init_params() {
  auto stage_size = bmrt_get_stage_size(p_bmrt, "block_0");
  if (stage_idx < 0 || stage_idx >= stage_size) {
    throw std::runtime_error("Invalid stage idx");
  }

  // read parameters from bmodel
  is_dynamic = net_blocks[0]->is_dynamic;
  hidden_bytes = bm_mem_get_device_size(
      net_blocks_cache[0]->stages[stage_idx].output_mems[0]);
  kv_bytes = bm_mem_get_device_size(
      net_blocks_cache[0]->stages[stage_idx].output_mems[1]);
  MAX_PREFILL_LENGTH = net_blocks[0]->stages[stage_idx].input_shapes[0].dims[1];
  SEQLEN = net_blocks_cache[0]->stages[stage_idx].input_shapes[3].dims[1];

  // resize
  past_key.clear();
  past_value.clear();
  total_tokens.clear();

  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  total_tokens.resize(SEQLEN);

  // declare tmemory location for kvcache
  for (int i = 0; i < NUM_LAYERS; i++) {
    ASSERT(net_blocks_cache[i]->addr_mode == 1);
    past_key[i] = net_blocks_cache[i]->stages[stage_idx].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[stage_idx].input_mems[4];
  }
}

void Qwen::make_in_tensors() {
  if (make_in_tensors_flag){
    return;
  }

  bool ret = false;
  ret = bmrt_tensor_ex(&inputs_pid, p_bmrt, net_blocks[0]->input_loc_devices[1],
                       net_blocks[0]->input_dtypes[1],
                       net_blocks[0]->stages[stage_idx].input_shapes[1]);
  ASSERT(true == ret);

  ret = bmrt_tensor_ex(&inputs_attention, p_bmrt,
                       net_blocks[0]->input_loc_devices[2],
                       net_blocks[0]->input_dtypes[2],
                       net_blocks[0]->stages[stage_idx].input_shapes[2]);
  ASSERT(true == ret);

  ret = bmrt_tensor_ex(&next_pid, p_bmrt,
                       net_blocks_cache[0]->input_loc_devices[1],
                       net_blocks_cache[0]->input_dtypes[1],
                       net_blocks_cache[0]->stages[stage_idx].input_shapes[1]);
  ASSERT(true == ret);

  ret = bmrt_tensor_ex(&next_attention, p_bmrt,
                       net_blocks_cache[0]->input_loc_devices[2],
                       net_blocks_cache[0]->input_dtypes[2],
                       net_blocks_cache[0]->stages[stage_idx].input_shapes[2]);
  ASSERT(true == ret);

  make_in_tensors_flag = true;
}

void Qwen::init(const std::vector<int> &devices, const std::string &model_path,
                bool read_bmodel) {
  if (read_bmodel) {
    // step1 : load bmodel
    load_bmodel(devices, model_path);

    // step2 : init nets
    init_nets();
  }

  // step3 : init parameters
  init_params();

  // step4 : make in tensors
  make_in_tensors();
}

void Qwen::update_bmodel_weight(
  const std::string &model_path, const std::string &lora_path, const std::string &net_idx,
  const std::string &mem_idx, const std::vector<std::string> &weight_idx) {

  std::vector<const char*> weight_idx_cstr;
  weight_idx_cstr.reserve(weight_idx.size());

  for (const auto &idx : weight_idx) {
    weight_idx_cstr.push_back(idx.c_str());
  }

  auto ret = bmrt_update_bmodel_weight_with_decrypt(
    p_bmrt, model_path.c_str(), lora_path.c_str(), net_idx.c_str(), mem_idx.c_str(),
    weight_idx_cstr.data(), static_cast<int>(weight_idx_cstr.size()), decrypt_func_
  );

  ASSERT(true == ret, "load lora binary weight error");
}


void Qwen::empty_bmodel_weight(
  const std::string &model_path, const std::string &net_idx,
  const std::string &mem_idx, const std::vector<std::string> &weight_idx) {

  std::vector<const char*> weight_idx_cstr;
  weight_idx_cstr.reserve(weight_idx.size());

  for (const auto &idx : weight_idx) {
      weight_idx_cstr.push_back(idx.c_str());
  }

  auto ret = bmrt_empty_bmodel_weight_with_decrypt(
    p_bmrt, model_path.c_str(), net_idx.c_str(), mem_idx.c_str(),
    weight_idx_cstr.data(), static_cast<int>(weight_idx_cstr.size()), decrypt_func_
  );

  ASSERT(true == ret, "load lora binary weight error");
}

void Qwen::free_in_tensors() {
  bm_free_device(bm_handle, inputs_pid.device_mem);
  bm_free_device(bm_handle, inputs_attention.device_mem);
  bm_free_device(bm_handle, next_pid.device_mem);
  bm_free_device(bm_handle, next_attention.device_mem);
  make_in_tensors_flag = false;
}

int Qwen::greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];
  head_launch(net, logits_mem, 0);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

int Qwen::penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem,
                         std::vector<int> &input_tokens, int &token_length) {
  auto &in1_mem = net->stages[stage_idx].input_mems[1];
  auto &in2_mem = net->stages[stage_idx].input_mems[2];
  auto &in3_mem = net->stages[stage_idx].input_mems[3];
  auto &in4_mem = net->stages[stage_idx].input_mems[4];
  auto &out0_mem = net->stages[stage_idx].output_mems[0];
  auto &out1_mem = net->stages[stage_idx].output_mems[1];

  // repeat_penalty + top_p + top_k + temperature
  std::vector<int> generated_tokens(SEQLEN, input_tokens[token_length - 1]);
  repeat_last_n = std::min(repeat_last_n, token_length);
  std::copy(input_tokens.begin() + token_length - repeat_last_n,
            input_tokens.begin() + token_length, generated_tokens.begin());
  bm_memcpy_s2d(bm_handle, in1_mem, (void *)generated_tokens.data());
  bm_memcpy_s2d(bm_handle, in2_mem, (void *)&top_p);
  bm_memcpy_s2d(bm_handle, in3_mem, (void *)&temperature);
  bm_memcpy_s2d(bm_handle, in4_mem, (void *)&repeat_penalty);

  // inference
  head_launch(net, logits_mem, stage_idx);

  // get logit & token
  int candidate_num = net->stages[stage_idx].output_shapes[0].dims[1];
  std::vector<float> probs(candidate_num);
  bm_memcpy_d2s(bm_handle, probs.data(), out0_mem);
  std::vector<int> tokens(candidate_num);
  bm_memcpy_d2s(bm_handle, tokens.data(), out1_mem);

  // penalty_sample
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return tokens[dist(sgen)];
}

std::vector<uint16_t>
Qwen::load_and_infer_embedding(const std::vector<int> &tokens) {
  std::ifstream file(embedding_path, std::ios::binary);
  if (!file || !file.is_open()) {
    throw std::runtime_error("Unable to open file\n");
  }

  file.seekg(0, std::ios::end);
  uint64_t file_size_in_disk = file.tellg();
  if (file_size_in_disk < header_size) {
    throw std::runtime_error("file is too small\n");
  }

  uint64_t file_size_in_header = 0;
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char *>(&file_size_in_header), sizeof(uint64_t));
  if (file_size_in_disk != file_size_in_header) {
    throw std::runtime_error("Error: file size is not equal to file size in header\n");
  }

  size_t embedding_bytes = hidden_bytes;
  size_t embedding_dim = embedding_bytes / sizeof(uint16_t);
  int size = tokens.size();

  std::vector<uint16_t> buffer(size * embedding_dim);
  for (int i = 0; i < std::min(size, total_length); i++) {
    long long start_position = (long long)tokens[i] * embedding_bytes + header_size;
    file.seekg(start_position, std::ios::beg);
    if (file.fail()) {
      throw std::runtime_error("File size is not correct\n");
    }
    file.read(reinterpret_cast<char *>(&buffer[i * embedding_dim]),
              embedding_bytes);
    if (file.fail()) {
      throw std::runtime_error("File read failed\n");
    }
  }
  return buffer;
}

bm_device_mem_t Qwen::embedding_launch(const bm_net_info_t *net0,
                                       const bm_net_info_t *net1,
                                       const bm_net_info_t *net2,
                                       const std::vector<int> &tokens) {
  // embedding : net0->stages[stage_idx]
  // embedding_cache : net0->stages[0]
  bm_device_mem_t out_mem;
  if (!embedding_path.empty() && lora_embedding_flag != -1 && enable_lora_embedding) 
  {
    int this_stage_idx = (strcmp(net2->name, "lora_embedding") == 0) ? stage_idx : 0;
    auto &in0_mem = net2->stages[this_stage_idx].input_mems[0];
    auto &in1_mem = net2->stages[this_stage_idx].input_mems[1];
    out_mem = net2->stages[this_stage_idx].output_mems[0];
    empty(bm_handle, out_mem);

    bm_memcpy_s2d(bm_handle, in0_mem, (void *)tokens.data());
    auto buffer = load_and_infer_embedding(tokens);
    bm_memcpy_s2d(bm_handle, in1_mem, (void *)buffer.data());

    net_launch(net2, this_stage_idx); // prefil embedding
  } else if (embedding_path.empty()) 
  {
    int this_stage_idx = (strcmp(net0->name, "embedding") == 0) ? stage_idx : 0;
    auto &in_mem = net0->stages[this_stage_idx].input_mems[0];
    out_mem = net0->stages[this_stage_idx].output_mems[0];
    bm_memcpy_s2d(bm_handle, in_mem, (void *)tokens.data());
    net_launch(net0, this_stage_idx); // prefil embedding
  } else if (!embedding_path.empty())
  {
    out_mem = net1->stages[stage_idx].input_mems[0];
    empty(bm_handle, out_mem);
    auto buffer = load_and_infer_embedding(tokens);
    bm_memcpy_s2d(bm_handle, out_mem, (void *)buffer.data());
  } else {
    throw std::runtime_error("embedding launch error");
  }
  return out_mem;
}

bm_device_mem_t Qwen::lm_launch(const bm_net_info_t *net,
                                const bm_device_mem_t &out_mem, size_t offset,
                                size_t size) {
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem, offset, size);
  net_launch(net_lm, 0);
  return lm_out_mem;
}

int Qwen::forward_first(std::vector<int> &tokens) {
  if ((int)tokens.size() >= MAX_PREFILL_LENGTH) {
    throw std::runtime_error("the sequence length you input exceeds MAX_PREFILL_LENGTH");
  }
  std::vector<int> first_tokens(MAX_PREFILL_LENGTH, 0);
  std::vector<int> position_id(MAX_PREFILL_LENGTH, 0);
  std::vector<uint16_t> attention_mask(MAX_PREFILL_LENGTH * MAX_PREFILL_LENGTH,
                                       mask_value);
  // std::fill(total_tokens.begin(), total_tokens.end(), 0);
  std::copy(tokens.begin(), tokens.end(), total_tokens.data());
  std::copy(tokens.begin(), tokens.end(), first_tokens.data());

  total_length = tokens.size();
  prefill_length = 0;

  for (int i = 0; i < total_length; i++) {
    position_id[i] = i;
  }
  for (int i = 0; i < total_length; i++) {
    for (int j = 0; j < MAX_PREFILL_LENGTH; j++) {
      if (j <= i) {
        attention_mask[i * MAX_PREFILL_LENGTH + j] = 0;
      }
    }
  }

  // empty
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty_net(bm_handle, net_blocks[i], stage_idx);
    empty_net(bm_handle, net_blocks_cache[i], stage_idx);
  }

  // forward embeding
  auto out_mem = embedding_launch(net_embed, net_blocks[0], net_lora_embed, first_tokens);

  // forward blocks
  // make in tensors
  bm_memcpy_s2d(bm_handle, inputs_pid.device_mem, (void *)position_id.data());
  bm_memcpy_s2d(bm_handle, inputs_attention.device_mem,
                (void *)attention_mask.data());
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    // init
    auto &in0_mem = net_blocks[idx]->stages[stage_idx].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[stage_idx].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[stage_idx].input_mems[2];

    // move to device
    d2d(in0_mem, out_mem, 0, total_length * hidden_bytes);
    in1_mem = inputs_pid.device_mem;
    in2_mem = inputs_attention.device_mem;

    // net forward
    // can not to dynamic net launch for combine qwen2-10240 and qwen2-5120
    // if (net_blocks[idx]->is_dynamic) {
    //   dynamic_net_launch(net_blocks[idx], total_length, stage_idx);
    // } else {
    //   net_launch(net_blocks[idx], stage_idx);
    // }
    net_launch(net_blocks[idx], stage_idx);
    out_mem = net_blocks[idx]->stages[stage_idx].output_mems[0];
    d2d(past_key[idx], net_blocks[idx]->stages[stage_idx].output_mems[1], 0,
        total_length * kv_bytes);
    d2d(past_value[idx], net_blocks[idx]->stages[stage_idx].output_mems[2], 0,
        total_length * kv_bytes);
  }

  // forward lmhead
  auto lm_out_mem = lm_launch(net_lm, out_mem,
                              (total_length - 1) * hidden_bytes, hidden_bytes);

  int token = 0;
  if (generation_mode == "greedy") {
    token = greedy_search(net_greedy_head, lm_out_mem);
  } else if (generation_mode == "penalty_sample") {
    token = penalty_sample(net_penalty_sample_head, lm_out_mem, total_tokens,
                           total_length);
  }

  total_tokens[total_length] = token;
  total_length += 1;
  return token;
}

int Qwen::forward_next() {
  if (total_length >= SEQLEN - 5) {
    throw std::runtime_error("the sequence length exceeds SEQLEN");
  }

  int cur_token = total_tokens[total_length - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = total_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = mask_value;
  }
  int32_t position_id = total_length - 1;

  // embedding
  std::vector<int> cur_tokens = {cur_token};
  auto out_mem =
      embedding_launch(net_embed_cache, net_blocks_cache[0], net_lora_embed_cache, cur_tokens);

  // blocks
  // move psition_id & attention_mask to device
  bm_memcpy_s2d(bm_handle, next_pid.device_mem, &position_id);
  bm_memcpy_s2d(bm_handle, next_attention.device_mem,
                (void *)attention_mask.data());
  int token_offset = (total_length - 1) * kv_bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    // init
    auto &in0_mem = net_blocks_cache[idx]->stages[stage_idx].input_mems[0];
    auto &in1_mem = net_blocks_cache[idx]->stages[stage_idx].input_mems[1];
    auto &in2_mem = net_blocks_cache[idx]->stages[stage_idx].input_mems[2];
    auto &out0_mem = net_blocks_cache[idx]->stages[stage_idx].output_mems[0];
    auto &out1_mem = net_blocks_cache[idx]->stages[stage_idx].output_mems[1];
    auto &out2_mem = net_blocks_cache[idx]->stages[stage_idx].output_mems[2];

    // move to device
    // empty(bm_handle, in0_mem);
    d2d(in0_mem, out_mem);
    in1_mem = next_pid.device_mem;
    in2_mem = next_attention.device_mem;

    // net forward
    net_launch(net_blocks_cache[idx], stage_idx);

    out_mem = out0_mem;
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], token_offset, out1_mem, 0,
                       kv_bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], token_offset, out2_mem, 0,
                       kv_bytes);
  }
  // forward lmhead
  auto lm_out_mem = lm_launch(net_lm, out_mem, 0, hidden_bytes);

  int token = 0;
  if (generation_mode == "greedy") {
    token = greedy_search(net_greedy_head, lm_out_mem);
  } else if (generation_mode == "penalty_sample") {
    token = penalty_sample(net_penalty_sample_head, lm_out_mem, total_tokens,
                           total_length);
  }
  
  total_tokens[total_length] = token;
  total_length += 1;
  return token;
}

// std::vector<int> Qwen::generate(std::vector<int> &history_tokens, int EOS) {
//   if (history_tokens.empty(bm_handle, )) {
//     printf("Sorry: your question is empty!!\n");
//     history_tokens.clear();
//     return {};
//   }

//   // make sure token not too large
//   if ((int)history_tokens.size() > SEQLEN - 10) {
//     history_tokens.clear();
//     printf("Error: your question is too large!\n");
//     return {};
//   }

//   std::vector<int> result_tokens;
//   int token = forward_first(history_tokens);
//   while (token != EOS && token_length < SEQLEN) {
//     result_tokens.emplace_back(token);
//     token = forward_share_next();
//   }

//   return result_tokens;
// }

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Qwen>(m, "Qwen")
      .def(pybind11::init<>())
      .def("init", &Qwen::init)
      .def("forward_first", &Qwen::forward_first)
      .def("forward_next", &Qwen::forward_next)
      .def("deinit", &Qwen::deinit)
      .def("init_decrypt", &Qwen::init_decrypt)
      .def("deinit_decrypt", &Qwen::deinit_decrypt)
      .def("update_bmodel_weight", &Qwen::update_bmodel_weight)
      .def("empty_bmodel_weight", &Qwen::empty_bmodel_weight)
      .def_readwrite("SEQLEN", &Qwen::SEQLEN) // read SEQLEN in pipeline.py
      .def_readwrite("MAX_PREFILL_LENGTH", &Qwen::MAX_PREFILL_LENGTH)
      .def_readwrite("total_length", &Qwen::total_length)
      .def_readwrite("prefill_length", &Qwen::prefill_length)
      .def_readwrite("temperature", &Qwen::temperature)
      .def_readwrite("top_p", &Qwen::top_p)
      .def_readwrite("repeat_penalty", &Qwen::repeat_penalty)
      .def_readwrite("repeat_last_n", &Qwen::repeat_last_n)
      .def_readwrite("max_new_tokens", &Qwen::max_new_tokens)
      .def_readwrite("generation_mode", &Qwen::generation_mode)
      .def_readwrite("prefill_reuse", &Qwen::prefill_reuse)
      .def_readwrite("status_code", &Qwen::status_code)
      .def_readwrite("lib_path", &Qwen::lib_path)
      .def_readwrite("stage_idx", &Qwen::stage_idx)
      .def_readwrite("enable_lora_embedding", &Qwen::enable_lora_embedding)
      .def_readwrite("embedding_path", &Qwen::embedding_path);
}
