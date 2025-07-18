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
#include <pybind11/numpy.h>
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
  void init(const std::vector<int> &devid, const std::string &model_path);
  void deinit();
  void init_decrypt();
  void deinit_decrypt();

  // Inference Functions
  void update_config();
  void init_forward(pybind11::array_t<int> tokens);
  int forward_first();
  int forward_next();
  std::vector<int> generate(const std::vector<int> &EOS);

  // Media Processing
  void process_media(const std::string &media_path,
                     const std::string &media_type,
                     pybind11::array_t<float> pixel_values_arr);

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

  // Core Inference Functions
  void vit_launch(const std::vector<float> &pixel_values, int vit_offset,
                  int vit_size);
  void net_launch(const bm_net_info_t *net, int stage = 0);
  void dynamic_net_launch(const bm_net_info_t *net, int token_length,
                          int stage_idx = 0);
  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem,
                   int this_stage_idx);
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
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, size_t offset,
                  size_t size);

public:
  bool is_dynamic;
  uint32_t prefill_reuse;
  std::vector<int> raw_tokens;
  std::vector<int> total_tokens;
  std::string lib_path;
  std::string embedding_path;
  int stage_idx;
  bool make_in_tensors_flag;

  // model
  Config config;
  int hidden_bytes;
  int kv_bytes;
  int total_length;
  int SEQLEN;
  int NUM_LAYERS;
  int MAX_PREFILL_LENGTH;
  int BATCH_SIZE;
  std::unique_ptr<Maker> maker;

  // vit config
  int MAX_PIXELS;
  int VIT_DIMS;
  std::vector<int> media_offset;
  std::vector<int> media_size;

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
  void *decrypt_handle_;      // handle of decrypt lib
  decrypt_func decrypt_func_; // decrypt func from lib
};

// init
Model::Model() {
  prefill_reuse = 0;
  stage_idx = 0;
  raw_tokens.clear();
  total_tokens.clear();
  make_in_tensors_flag = false;

  // path
  // lib_path = "";
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
  decrypt_handle_ = nullptr;
  decrypt_func_ = nullptr;
}

void Model::deinit() {

  bm_free_device(bm_handle, dev_buffer);

  if (p_bmrt != nullptr) {
    bmrt_destroy(p_bmrt);
    p_bmrt = nullptr;
  }

  bm_dev_free(bm_handle);
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

void Model::init_decrypt() {
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

void Model::deinit_decrypt() {
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
    vision_enabled = false; // only when vit launch, vision_enabled = true
    net_vit = bmrt_get_network_info(p_bmrt, "vit");
    MAX_PIXELS = net_vit->stages[0].input_shapes[0].dims[0];
    VIT_DIMS = net_vit->stages[0].input_shapes[0].dims[1];
    auto status = bm_malloc_device_byte(
        bm_handle, &dev_buffer,
        bm_mem_get_device_size(net_vit->stages[0].output_mems[0]));
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

  // read parameters from bmodel
  is_dynamic = net_blocks[0]->is_dynamic;
  hidden_bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[0]);
  kv_bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  MAX_PREFILL_LENGTH = net_blocks[0]->stages[1].input_shapes[0].dims[1];
  SEQLEN = net_blocks_cache[0]->stages[1].input_shapes[3].dims[1];

  // resize
  past_key.clear();
  past_value.clear();
  raw_tokens.clear();
  total_tokens.clear();

  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  total_tokens.resize(SEQLEN);
}

void Model::init(const std::vector<int> &devices,
                 const std::string &model_path) {
  // step1 : load bmodel
  load_bmodel(devices, model_path);

  // step2 : init nets
  init_network();

  // step3 : init parameters
  init_parameter();
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

#ifdef ENABLE_MEDIA
#include "cv_utils.h"
void Model::process_media(
    const std::string &media_path, const std::string &media_type,
    pybind11::array_t<float> pixel_values_arr = pybind11::array_t<float>()) {
  int media_token_id;
  std::vector<float> pixel_values;

  if (pixel_values_arr.size() != 0) {
    pybind11::buffer_info buf = pixel_values_arr.request();
    float *ptr = static_cast<float *>(buf.ptr);
    size_t size = buf.size;
    pixel_values.assign(ptr, ptr + size);
  } else {
    if (media_type == "image") {
      pixel_values = process_image(media_path, config);
    } else if (media_type == "video") {
      process_video(media_path);
    } else if (media_type == "audio") {
      process_audio(media_path);
    } else {
      throw std::runtime_error("not support now");
    }
  }

  if (media_type == "image") {
    media_token_id = config.image_token_id;
  } else if (media_type == "video") {
    media_token_id = config.video_token_id;
  }

  // token process & vit launch
  raw_tokens = maker->insert_tokens(raw_tokens, media_token_id);
  get_media_info(raw_tokens, media_offset, media_size, media_token_id);

  ASSERT((media_offset.size() == 1 && media_size.size() == 1),
         "only support media_offset.size() == 1");

  // update
  total_length = raw_tokens.size();
  config.total_length = total_length;
  config.media_offset = media_offset[0];
  config.media_size = media_size[0];

  // vit launch
  vit_launch(pixel_values, media_offset[0], media_size[0]);
}
#endif

void Model::vit_launch(const std::vector<float> &pixel_values, int vit_offset,
                       int vit_size) {
  auto start = std::chrono::high_resolution_clock::now();
  empty(bm_handle, dev_buffer);

  // forward vision transformer
  std::vector<float> pixel_values_pad(MAX_PIXELS * VIT_DIMS, 0);
  auto position_id = maker->make_vit_position_id();
  auto attention_mask = maker->make_vit_attention_mask();
  std::copy(pixel_values.begin(), pixel_values.end(), pixel_values_pad.data());
  empty_net(bm_handle, net_vit);

  auto &vit_in0_mem = net_vit->stages[0].input_mems[0];
  auto &vit_in1_mem = net_vit->stages[0].input_mems[1];
  auto &vit_in2_mem = net_vit->stages[0].input_mems[2];
  auto &vit_out_mem = net_vit->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, vit_in0_mem, (void *)pixel_values_pad.data());
  bm_memcpy_s2d(bm_handle, vit_in1_mem, (void *)position_id.data());
  bm_memcpy_s2d(bm_handle, vit_in2_mem, (void *)attention_mask.data());
  net_launch(net_vit, 0);

  int vit_bytes = vit_size * hidden_bytes;
  bm_memcpy_d2d_byte(bm_handle, dev_buffer, 0, vit_out_mem, 0, vit_bytes);

  vision_enabled = true;
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "vit_launch time : " << duration.count() << " milliseconds."
            << std::endl;
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

void Model::update_config() {
  config.SEQLEN = SEQLEN;
  config.MAX_PREFILL_LENGTH = MAX_PREFILL_LENGTH;
  config.total_length = total_length;

  config.max_pos = 0;
  config.MAX_PIXELS = MAX_PIXELS;
}

void Model::init_forward(pybind11::array_t<int> tokens) {
  pybind11::buffer_info buf = tokens.request();
  int *ptr = static_cast<int *>(buf.ptr);
  size_t size = buf.size;
  size_t prefill_length_0 = net_blocks[0]->stages[0].input_shapes[0].dims[1];
  size_t prefill_length_1 = net_blocks[0]->stages[1].input_shapes[0].dims[1];
  size_t prefill_length_2 = net_blocks[0]->stages[2].input_shapes[0].dims[1];
  if (size > prefill_length_1) {
    stage_idx = 2;
    MAX_PREFILL_LENGTH = prefill_length_2;
    SEQLEN = net_blocks_cache[0]->stages[2].input_shapes[3].dims[1];
  } else if (size > prefill_length_0) {
    stage_idx = 1;
    MAX_PREFILL_LENGTH = prefill_length_1;
    SEQLEN = net_blocks_cache[0]->stages[1].input_shapes[3].dims[1];
  } else {
    stage_idx = 0;
    MAX_PREFILL_LENGTH = prefill_length_0;
    SEQLEN = net_blocks_cache[0]->stages[0].input_shapes[3].dims[1];
  }
  std::cout << "use stage_idx : " << stage_idx << "  input tokens : " << size
            << std::endl;
  for (int i = 0; i < NUM_LAYERS; i++) {
    ASSERT(net_blocks_cache[i]->addr_mode == 1, "");
    past_key[i] = net_blocks_cache[i]->stages[stage_idx].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[stage_idx].input_mems[4];
  }

  raw_tokens.resize(size);
  memcpy(raw_tokens.data(), ptr, size * sizeof(int));

  total_length = raw_tokens.size();
  update_config();
  maker = std::make_unique<Maker>(config);
}

int Model::forward_first() {
  ASSERT((int)raw_tokens.size() < MAX_PREFILL_LENGTH,
         "the sequence length you input exceeds MAX_PREFILL_LENGTH");

  std::vector<int> first_tokens(MAX_PREFILL_LENGTH, 0);
  std::copy(raw_tokens.begin(), raw_tokens.end(), total_tokens.data());
  std::copy(raw_tokens.begin(), raw_tokens.end(), first_tokens.data());

  auto position_id = maker->make_position_id();
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, mask_value);
  for (int i = 0; i < total_length; i++) {
    for (int j = 0; j <= i; j++) {
      attention_mask[i * SEQLEN + j] = 0;
    }
  }

  // empty
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty_net(bm_handle, net_blocks[i], stage_idx);
    empty_net(bm_handle, net_blocks_cache[i], stage_idx);
  }

  // forward embeding
  auto out_mem = embedding_launch(net_embed, net_blocks[0], first_tokens);

  // forward vit
  if (vision_enabled) {
    for (size_t i = 0; i < media_offset.size(); i++) {
      bm_memcpy_d2d_byte(bm_handle, out_mem, media_offset[i] * hidden_bytes,
                         dev_buffer, 0, media_size[i] * hidden_bytes);
    }
  }

  // forward blocks
  // make in tensors
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    // init
    auto &in0_mem = net_blocks[idx]->stages[stage_idx].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[stage_idx].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[stage_idx].input_mems[2];

    // move to device
    d2d(in0_mem, out_mem, 0, total_length * hidden_bytes);
    bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
    bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());

    // net forward
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

  int cur_token = total_tokens[total_length - 1];

  config.total_length = total_length;
  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = total_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = mask_value;
  }
  auto position_id = maker->make_next_position_id();

  // embedding
  std::vector<int> cur_tokens = {cur_token};
  auto out_mem =
      embedding_launch(net_embed_cache, net_blocks_cache[0], cur_tokens);

  // blocks
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
    bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
    bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());

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

std::vector<int> Model::generate(const std::vector<int> &EOS) {
  std::vector<int> result_tokens;
  int token = forward_first();
  while (std::find(EOS.begin(), EOS.end(), token) == EOS.end() &&
         total_length < SEQLEN && (int)result_tokens.size() < max_new_tokens) {
    result_tokens.emplace_back(token);
    token = forward_next();
  }
  return result_tokens;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Config>(m, "Config")
      .def(pybind11::init<>())
      .def_readwrite("model_type", &Config::model_type)
      .def_readwrite("grid_thw", &Config::grid_thw)
      .def_readwrite("patch_size", &Config::patch_size)
      .def_readwrite("spatial_merge_size", &Config::spatial_merge_size)
      .def_readwrite("temporal_patch_size", &Config::temporal_patch_size)
      .def_readwrite("resized_height", &Config::resized_height)
      .def_readwrite("resized_width", &Config::resized_width)
      .def_readwrite("image_token_id", &Config::image_token_id)
      .def_readwrite("video_token_id", &Config::video_token_id);

  pybind11::class_<Model>(m, "Model")
      .def(pybind11::init<>())
      .def("init", &Model::init)
      .def("init_forward", &Model::init_forward)
      .def("forward_first", &Model::forward_first)
      .def("forward_next", &Model::forward_next)
#ifdef ENABLE_MEDIA
      .def("process_media", &Model::process_media, pybind11::arg("media_path"),
           pybind11::arg("media_type"),
           pybind11::arg("pixel_values_arr") = pybind11::array_t<float>())
#endif
      .def("generate", &Model::generate)
      .def("deinit", &Model::deinit)
      .def("init_decrypt", &Model::init_decrypt)
      .def("deinit_decrypt", &Model::deinit_decrypt)
      .def_readwrite("config", &Model::config)
      .def_readwrite("SEQLEN", &Model::SEQLEN)
      .def_readwrite("NUM_LAYERS", &Model::NUM_LAYERS)
      .def_readwrite("MAX_PREFILL_LENGTH", &Model::MAX_PREFILL_LENGTH)
      .def_readwrite("MAX_PIXELS", &Model::MAX_PIXELS)
      .def_readwrite("total_length", &Model::total_length)
      .def_readwrite("temperature", &Model::temperature)
      .def_readwrite("top_p", &Model::top_p)
      .def_readwrite("repeat_penalty", &Model::repeat_penalty)
      .def_readwrite("repeat_last_n", &Model::repeat_last_n)
      .def_readwrite("max_new_tokens", &Model::max_new_tokens)
      .def_readwrite("generation_mode", &Model::generation_mode)
      .def_readwrite("prefill_reuse", &Model::prefill_reuse)
      .def_readwrite("lib_path", &Model::lib_path)
      .def_readwrite("stage_idx", &Model::stage_idx)
      .def_readwrite("embedding_path", &Model::embedding_path);
}
