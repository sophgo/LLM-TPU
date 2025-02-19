//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
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

class Model {
public:
  // Initialization
  void init(const std::vector<int> &devid, const std::string &model_path,
            bool read_bmodel = true);
  void deinit();

  // Inference Functions
  int forward_first(const std::vector<int> &tokens,
                    const std::vector<float> &pixel_values = {},
                    const std::vector<int> &grid_thw = {}, int vit_offset = 0,
                    int valid_vit_length = 0);
  int forward_next();
  std::vector<int> generate(std::vector<int> &history_tokens, int EOS);

  // Image Processing
  void process_image(const std::string &image_path);

  std::mt19937 sgen;
  bool vision_enabled = false;
  Model();
  ~Model();

private:
  // Internal Utilities
  void load_bmodel(const std::vector<int> &devices,
                   const std::string &model_path);
  void init_network();
  void init_parameter();
  void make_in_tensors();
  void free_in_tensors();

  // Core Inference Functions
  void vit_launch(const std::vector<float> &pixel_values, int vit_offset,
                  int valid_vit_length, const std::vector<int> &grid_thw,
                  bm_device_mem_t &out_mem);
  void net_launch(const bm_net_info_t *net, int stage = 0);
  void dynamic_net_launch(const bm_net_info_t *net, int token_length,
                          int stage_idx = 0);
  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem,
                   int this_stage_idx);
  int sample_token(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  std::vector<uint16_t>
  load_and_infer_embedding(const std::vector<int> &tokens);
  bm_device_mem_t embedding_launch(const bm_net_info_t *net0,
                                   const bm_net_info_t *net1,
                                   const std::vector<int> &tokens);
  bm_device_mem_t lm_launch(const bm_net_info_t *net,
                            const bm_device_mem_t &out_mem, size_t offset,
                            size_t size);

  // Sample Functions
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem,
                     std::vector<int> &input_tokens, int &token_length);

  // Helper Functions
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, size_t offset);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, size_t offset,
                  size_t size);
  std::vector<int> make_posid(const std::vector<int> &grid_thw, int vit_offset,
                              int valid_vit_length, int token_length);
  std::vector<uint16_t> create_attention_mask(int seq_len);

public:
  bool is_dynamic;
  uint32_t prefill_reuse;
  std::vector<int> total_tokens;
  std::string embedding_path;
  int stage_idx;
  bool make_in_tensors_flag;

  // model
  Config config;
  std::string model_type;
  int hidden_bytes;
  int kv_bytes;
  int total_length;
  int SEQLEN;
  int NUM_LAYERS;
  int MAX_PREFILL_LENGTH;
  int BATCH_SIZE;

  // vit config
  int MAX_PIXELS;
  int VIT_DIMS;
  int spatial_merge_size;

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
  const bm_net_info_t *net_lm, *net_greedy_head, *net_penalty_sample_head;
  const bm_net_info_t *net_vit;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
  bm_tensor_t inputs_pid, next_pid;
  bm_tensor_t inputs_attention, next_attention;

  uint16_t mask_value;
};

// init
Model::Model() {
  prefill_reuse = 0;
  stage_idx = 0;
  total_tokens.clear();
  make_in_tensors_flag = false;

  // path
  embedding_path = "";

  // length
  total_length = 0;
  SEQLEN = 0;
  NUM_LAYERS = 0;
  MAX_PREFILL_LENGTH = 0;
  MAX_PIXELS = 0;
  VIT_DIMS = 0;

  //
  sgen = std::mt19937(std::random_device()());
  bm_handle = nullptr;
  p_bmrt = nullptr;
}

void Model::deinit() {
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

Model::~Model() { deinit(); }

static inline void ASSERT(bool ret, std::string message) {
  if (!ret) {
    throw std::runtime_error(message);
  }
}

void Model::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, size_t offset,
                size_t size) {
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void Model::head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem,
                        int this_stage_idx) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  bmrt_tensor_with_device(&in_tensors[0], logits_mem, net->input_dtypes[0],
                          net->stages[this_stage_idx].input_shapes[0]);

  for (int i = 1; i < net->input_num; i++) {
    bmrt_tensor_with_device(
        &in_tensors[i], net->stages[this_stage_idx].input_mems[i],
        net->input_dtypes[i], net->stages[this_stage_idx].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(
        &out_tensors[i], net->stages[this_stage_idx].output_mems[i],
        net->output_dtypes[i], net->stages[this_stage_idx].output_shapes[i]);
  }

  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);

  ASSERT(ret == true, "can not inference bmodel");
  bm_thread_sync(bm_handle);
}

void Model::net_launch(const bm_net_info_t *net, int stage_idx) {
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

  ASSERT(ret == true, "can not inference bmodel");
  bm_thread_sync(bm_handle);
}

void Model::dynamic_net_launch(const bm_net_info_t *net, int token_length,
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

  ASSERT(ret == true, "can not inference bmodel");
  bm_thread_sync(bm_handle);
}

void Model::load_bmodel(const std::vector<int> &devices,
                        const std::string &model_path) {
  // Device Initialization
  ASSERT(devices.size() == 1, "not support multi device");
  std::cout << "Initializing devices...\n";
  std::cout << "Device [ " << devices[0] << " ] loading .....\n";
  bm_status_t status = bm_dev_request(&bm_handle, devices[0]);
  ASSERT(status == BM_SUCCESS, "can not create handle");

  // create bmruntime
  p_bmrt = bmrt_create(bm_handle);
  ASSERT(p_bmrt != NULL, "can not create bmrt");

  std::string board_name(256, '\0');
  bm_get_board_name(bm_handle, &board_name[0]);

  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());

  ASSERT(ret == true, "can not load bmodel correctly");
  printf("Done!\n");
}

void Model::init_network() {
  // net embed and lm_head
  ASSERT(bmrt_get_network_index(p_bmrt, "embedding") != -1 ||
             !embedding_path.empty(),
         "bmodel is lack of embedding or embedding_path is empty");
  if (embedding_path.empty()) {
    net_embed = bmrt_get_network_info(p_bmrt, "embedding");
    net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  }
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
  net_penalty_sample_head =
      bmrt_get_network_info(p_bmrt, "penalty_sample_head");

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

  // Vision Components
  if (bmrt_get_network_index(p_bmrt, "vit") != -1) {
    vision_enabled = true;
    net_vit = bmrt_get_network_info(p_bmrt, "vit");
    MAX_PIXELS = net_vit->stages[0].input_shapes[0].dims[0];
    VIT_DIMS = net_vit->stages[0].input_shapes[0].dims[1];
    auto status = bm_malloc_device_byte(
        bm_handle, &dev_buffer,
        bm_mem_get_device_size(net_blocks[0]->stages[0].input_mems[0]));
    ASSERT(status == BM_SUCCESS, "malloc memory failed");
  }

  // Mask Value Setup
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

void Model::init_parameter() {
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
    ASSERT(net_blocks_cache[i]->addr_mode == 1, "");
    past_key[i] = net_blocks_cache[i]->stages[stage_idx].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[stage_idx].input_mems[4];
  }
}

void Model::make_in_tensors() {
  if (make_in_tensors_flag) {
    return;
  }

  bool ret = false;
  ret = bmrt_tensor_ex(&inputs_pid, p_bmrt, net_blocks[0]->input_loc_devices[1],
                       net_blocks[0]->input_dtypes[1],
                       net_blocks[0]->stages[stage_idx].input_shapes[1]);
  ASSERT(true == ret, "malloc tensor failed");

  ret = bmrt_tensor_ex(&inputs_attention, p_bmrt,
                       net_blocks[0]->input_loc_devices[2],
                       net_blocks[0]->input_dtypes[2],
                       net_blocks[0]->stages[stage_idx].input_shapes[2]);
  ASSERT(true == ret, "malloc tensor failed");

  ret = bmrt_tensor_ex(&next_pid, p_bmrt,
                       net_blocks_cache[0]->input_loc_devices[1],
                       net_blocks_cache[0]->input_dtypes[1],
                       net_blocks_cache[0]->stages[stage_idx].input_shapes[1]);
  ASSERT(true == ret, "malloc tensor failed");

  ret = bmrt_tensor_ex(&next_attention, p_bmrt,
                       net_blocks_cache[0]->input_loc_devices[2],
                       net_blocks_cache[0]->input_dtypes[2],
                       net_blocks_cache[0]->stages[stage_idx].input_shapes[2]);
  ASSERT(true == ret, "malloc tensor failed");

  make_in_tensors_flag = true;
}

void Model::init(const std::vector<int> &devices, const std::string &model_path,
                 bool read_bmodel) {
  if (read_bmodel) {
    // step1 : load bmodel
    load_bmodel(devices, model_path);

    // step2 : init nets
    init_network();
  }

  // step3 : init parameters
  init_parameter();

  // step4 : make in tensors
  make_in_tensors();
}

void Model::free_in_tensors() {
  bm_free_device(bm_handle, inputs_pid.device_mem);
  bm_free_device(bm_handle, inputs_attention.device_mem);
  bm_free_device(bm_handle, next_pid.device_mem);
  bm_free_device(bm_handle, next_attention.device_mem);
  make_in_tensors_flag = false;
}

int Model::greedy_search(const bm_net_info_t *net,
                         bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];
  head_launch(net, logits_mem, 0);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

int Model::penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem,
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
Model::load_and_infer_embedding(const std::vector<int> &tokens) {
  std::ifstream file(embedding_path, std::ios::binary);
  if (!file || !file.is_open()) {
    throw std::runtime_error("Unable to open file\n");
  }

  size_t embedding_bytes = hidden_bytes;
  size_t embedding_dim = embedding_bytes / sizeof(uint16_t);
  int size = tokens.size();

  std::vector<uint16_t> buffer(size * embedding_dim);
  for (int i = 0; i < std::min(size, total_length); i++) {
    long long start_position = (long long)tokens[i] * embedding_bytes;
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

bm_device_mem_t Model::embedding_launch(const bm_net_info_t *net0,
                                        const bm_net_info_t *net1,
                                        const std::vector<int> &tokens) {
  // embedding : net0->stages[stage_idx]
  // embedding_cache : net0->stages[0]
  bm_device_mem_t out_mem;
  if (embedding_path.empty()) {
    int this_stage_idx = (strcmp(net0->name, "embedding") == 0) ? stage_idx : 0;
    auto &in_mem = net0->stages[this_stage_idx].input_mems[0];
    out_mem = net0->stages[this_stage_idx].output_mems[0];
    bm_memcpy_s2d(bm_handle, in_mem, (void *)tokens.data());
    net_launch(net0, this_stage_idx); // prefil embedding
  } else if (!embedding_path.empty()) {
    out_mem = net1->stages[stage_idx].input_mems[0];
    empty(bm_handle, out_mem);
    auto buffer = load_and_infer_embedding(tokens);
    bm_memcpy_s2d(bm_handle, out_mem, (void *)buffer.data());
  } else {
    throw std::runtime_error("embedding launch error");
  }
  return out_mem;
}

void Model::process_image(const std::string &image_path) {
  /**
  std::vector<cv::Mat> images;
  opencv_read_image(images, image_path);

  for (size_t i = 0; i < images.size(); i++) {
    auto image = images[0];
    int width = image.cols;
    int height = image.rows;
    if (model_type == "qwen2_vl") {
      std::vector<float> image_mean = {0.48145466f, 0.4578275f, 0.40821073f};
      std::vector<float> image_std = {0.26862954f, 0.26130258f, 0.27577711f};

      auto resized = smart_resize(height, width);
      int resized_height = resized.first;
      int resized_width = resized.second;
      auto resized_image = bicubic_resize(image, resized_height, resized_width, image_mean, image_std);
    }
  }

  **/
  return ;
}

void Model::vit_launch(const std::vector<float> &pixel_values, int vit_offset,
                       int valid_vit_length, const std::vector<int> &grid_thw,
                       bm_device_mem_t &out_mem) {
  auto start = std::chrono::high_resolution_clock::now();
  empty(bm_handle, dev_buffer);
  d2d(dev_buffer, out_mem, 0, total_length * hidden_bytes);
  out_mem = dev_buffer;
  // forward vision transformer
  std::vector<float> pixel_values_pad(MAX_PIXELS * VIT_DIMS, 0);
  auto position_id = make_vit_position_id(config);
  auto attention_mask = make_vit_attention_mask(config);
  std::copy(pixel_values.begin(), pixel_values.end(), pixel_values_pad.data());

  empty_net(bm_handle, net_vit);

  auto &vit_in0_mem = net_vit->stages[0].input_mems[0];
  auto &vit_in1_mem = net_vit->stages[0].input_mems[1];
  auto &vit_in2_mem = net_vit->stages[0].input_mems[2];
  auto &vit_out_mem = net_vit->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, vit_in0_mem, (void *)pixel_values_pad.data());
  bm_memcpy_s2d(bm_handle, vit_in1_mem, (void *)position_id.data());
  bm_memcpy_s2d(bm_handle, vit_in2_mem, (void *)attention_mask.data());
  net_launch(net_vit);

  // concatenante texting embedding and image embedding
  int dst_offset = vit_offset * hidden_bytes;
  int vit_size = valid_vit_length * hidden_bytes;
  bm_memcpy_d2d_byte(bm_handle, out_mem, dst_offset, vit_out_mem, 0, vit_size);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "vit_launch time : " << duration.count() << " milliseconds." << std::endl;
}

bm_device_mem_t Model::lm_launch(const bm_net_info_t *net,
                                 const bm_device_mem_t &out_mem, size_t offset,
                                 size_t size) {
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem, offset, size);
  net_launch(net_lm, 0);
  return lm_out_mem;
}

int Model::forward_first(const std::vector<int> &tokens,
                         const std::vector<float> &pixel_values,
                         const std::vector<int> &grid_thw, int vit_offset,
                         int valid_vit_length) {
  ASSERT((int)tokens.size() < MAX_PREFILL_LENGTH,
         "the sequence length you input exceeds MAX_PREFILL_LENGTH");

  std::vector<int> first_tokens(MAX_PREFILL_LENGTH, 0);
  std::copy(tokens.begin(), tokens.end(), total_tokens.data());
  std::copy(tokens.begin(), tokens.end(), first_tokens.data());

  total_length = tokens.size();

  config = {model_type,         SEQLEN,
            MAX_PREFILL_LENGTH, total_length,
            mask_value,         0,
            MAX_PIXELS,         grid_thw,
            vit_offset,         valid_vit_length,
            spatial_merge_size};
  auto position_id = make_position_id(config);
  auto attention_mask = make_attention_mask(config);

  // empty
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty_net(bm_handle, net_blocks[i], stage_idx);
    empty_net(bm_handle, net_blocks_cache[i], stage_idx);
  }

  // forward embeding
  auto out_mem = embedding_launch(net_embed, net_blocks[0], first_tokens);

  // forward vit
  if (vision_enabled && !pixel_values.empty()) {
    vit_launch(pixel_values, vit_offset, valid_vit_length, grid_thw, out_mem);
  }

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
    // can not to dynamic net launch for combine Model2-10240 and Model2-5120
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

int Model::forward_next() {
  if (total_length >= SEQLEN - 5) {
    throw std::runtime_error("the sequence length exceeds SEQLEN");
  }

  int cur_token = total_tokens[total_length - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = total_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = mask_value;
  }
  config.total_length = total_length;
  auto position_id = make_next_position_id(config);

  // embedding
  std::vector<int> cur_tokens = {cur_token};
  auto out_mem =
      embedding_launch(net_embed_cache, net_blocks_cache[0], cur_tokens);

  // blocks
  // move psition_id & attention_mask to device
  bm_memcpy_s2d(bm_handle, next_pid.device_mem, (void *)position_id.data());
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
    d2d(in0_mem, out_mem, 0, hidden_bytes);
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

// std::vector<int> Model::generate(std::vector<int> &history_tokens, int EOS) {
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
  pybind11::class_<Model>(m, "Model")
      .def(pybind11::init<>())
      .def("init", &Model::init)
      .def("forward_first", &Model::forward_first, pybind11::arg("tokens"),
           pybind11::arg("pixel_values") = std::vector<float>{},
           pybind11::arg("grid_thw") = std::vector<int>{},
           pybind11::arg("vit_offset") = 0,
           pybind11::arg("valid_vit_length") = 0)
      .def("forward_next", &Model::forward_next)
      .def("process_image", &Model::process_image)
      .def("deinit", &Model::deinit)
      .def_readwrite("model_type", &Model::model_type)
      .def_readwrite("SEQLEN", &Model::SEQLEN)
      .def_readwrite("NUM_LAYERS", &Model::NUM_LAYERS)
      .def_readwrite("MAX_PREFILL_LENGTH", &Model::MAX_PREFILL_LENGTH)
      .def_readwrite("MAX_PIXELS", &Model::MAX_PIXELS)
      .def_readwrite("spatial_merge_size", &Model::spatial_merge_size)
      .def_readwrite("total_length", &Model::total_length)
      .def_readwrite("temperature", &Model::temperature)
      .def_readwrite("top_p", &Model::top_p)
      .def_readwrite("repeat_penalty", &Model::repeat_penalty)
      .def_readwrite("repeat_last_n", &Model::repeat_last_n)
      .def_readwrite("max_new_tokens", &Model::max_new_tokens)
      .def_readwrite("generation_mode", &Model::generation_mode)
      .def_readwrite("prefill_reuse", &Model::prefill_reuse)
      .def_readwrite("stage_idx", &Model::stage_idx)
      .def_readwrite("embedding_path", &Model::embedding_path);
}
