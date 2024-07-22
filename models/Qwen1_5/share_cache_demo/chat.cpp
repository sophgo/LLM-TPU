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
#include <getopt.h>
#include <stdio.h>
#include <inttypes.h>
#include <random>
#include <numeric>
#include <fstream>
#include <cryptopp/osrng.h>
#include <cryptopp/aes.h>
#include <cryptopp/modes.h>
#include <cryptopp/filters.h>

#include "bmruntime_interface.h"
#include "memory.h"
#include "utils.h"
#include "crypto.h"


static const float ATTENTION_MASK = -10000.;
typedef struct {
  uint32_t magic;
  uint32_t header_size;
  uint32_t flatbuffers_size;
  uint32_t binary_size;
  uint32_t reserved[12];
} __attribute__((packed)) MODEL_HEADER_T;

class Qwen {
public:
  void init(const std::vector<int> &devid, std::string model_path);
  void deinit();
  void free_device();
  void malloc_bmodel_mem();
  void empty_kvcache();
  void forward_first(std::vector<int> &tokens);
  int forward_unshare(std::vector<int> &tokens);
  int forward_next();
  std::vector<int> generate(std::vector<int> &history_tokens, int EOS);

  void encrypt_bmodel(std::string model_path);
  std::vector<uint8_t> decrypt_bmodel(std::string model_path);

  std::mt19937 sgen;
  Qwen() : sgen(std::random_device()()){};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  void dynamic_net_launch(const bm_net_info_t *net, int token_length, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset, int size);

  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem, std::vector<int> &input_tokens, int &token_length);

  std::vector<uint8_t> read_file(std::string model_path, size_t size, size_t offset);
  std::vector<uint8_t> enc_file(std::string model_path, size_t size, size_t offset);
  std::vector<uint8_t> dec_file(std::string model_path, size_t size, size_t offset);

public:
  int share_length;
  int unshare_length;
  int SEQLEN;     // read from bmodel
  int NUM_LAYERS; // read from bmodel
  int MAX_SHARE_LENGTH;
  int MAX_UNSHARE_LENGTH;
  int BATCH_SIZE;
  bool io_alone;
  bool is_dynamic;
  bool memory_prealloc;
  bool is_decrypt;
  bool io_alone_reuse;
  std::vector<int> unshare_tokens;

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
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_unshare;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_unshare;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm, *net_greedy_head, *net_penalty_sample_head;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
  std::vector<bm_device_mem_t> prev_past_key;
  std::vector<bm_device_mem_t> prev_past_value;
  bm_device_mem_t tmp_key_cache;
  bm_device_mem_t tmp_value_cache;

  std::vector<bm_device_mem_u64_t> prealloc_mem_v;
  std::vector<bm_device_mem_t> io_mem_v;
  uint16_t mask_value;
  mem_info_t mem_info;
  AESOFBCipher cipher;
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

void Qwen::dynamic_net_launch(const bm_net_info_t *net, int token_length, int stage_idx) {
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
  // in_tensors[0].shape.dims[1] = token_length;
  // in_tensors[1].shape.dims[1] = token_length;
  // in_tensors[2].shape.dims[2] = token_length;
  // in_tensors[2].shape.dims[3] = token_length;
  // out_tensors[0].shape.dims[1] = token_length;
  // out_tensors[1].shape.dims[1] = token_length;
  // out_tensors[2].shape.dims[1] = token_length;
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

void Qwen::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(dst));
}

void Qwen::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset) {
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, bm_mem_get_device_size(src));
}

void Qwen::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset, int size) {
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void malloc_device_mem(bm_handle_t bm_handle, memory_t &mem, std::vector<bm_device_mem_u64_t> &prealloc_mem_v) {
  if (mem.size > 0) {
    bm_device_mem_u64_t dmem;
    if (bm_malloc_device_byte_u64(bm_handle, &dmem, mem.size) == BM_SUCCESS) {
       mem.addr = dmem.u.device.device_addr;
       prealloc_mem_v.push_back(dmem);
    }
  }
}

void Qwen::init(const std::vector<int> &devices, std::string model_path) {
  
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

  bool ret = false;
  std::vector<uint8_t> decrypted_data;

  // decrypt bmodel
  if (is_decrypt) {
    decrypted_data = decrypt_bmodel(model_path);
    ret= true;
  } else {
    ret = bmrt_get_bmodel_info(model_path.c_str(), &mem_info);
  }
  assert(true == ret);

  // prealloc memory
  if (memory_prealloc) {
    malloc_bmodel_mem();
    ret = bmrt_load_bmodel_with_mem_v2(p_bmrt, model_path.c_str(), (void*)decrypted_data.data(), &mem_info, io_mem_v);
  } else {
    ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  }
  assert(true == ret);
  printf("Done!\n");

  // net embed and lm_head
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_unshare = bmrt_get_network_info(p_bmrt, "embedding_unshare");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
  net_penalty_sample_head = bmrt_get_network_info(p_bmrt, "penalty_sample_head");
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = (num_nets - 5) / 3;

  // net blocks
  net_blocks.clear();
  net_blocks_unshare.clear();
  net_blocks_cache.clear();
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto unshare_name = "block_unshare_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_unshare.emplace_back(
        bmrt_get_network_info(p_bmrt, unshare_name.c_str()));
    net_blocks_cache.emplace_back(bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }

  MAX_SHARE_LENGTH = net_blocks[0]->stages[0].input_shapes[0].dims[1];
  MAX_UNSHARE_LENGTH = net_blocks_unshare[0]->stages[0].input_shapes[0].dims[1];
  SEQLEN = net_blocks_cache[0]->stages[0].input_shapes[3].dims[1];

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

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  prev_past_key.resize(NUM_LAYERS);
  prev_past_value.resize(NUM_LAYERS);

  is_dynamic = net_blocks[0]->is_dynamic;
  auto addr_mode = net_blocks_cache[0]->addr_mode;
  io_alone = addr_mode == 1;
  assert(io_alone == 1);
  
  if (io_alone_reuse) {
    ret = bm_malloc_device_byte(bm_handle, &tmp_key_cache, past_key[0].size);
    assert(BM_SUCCESS == ret);
    ret = bm_malloc_device_byte(bm_handle, &tmp_value_cache, past_value[0].size);
    assert(BM_SUCCESS == ret);
  }
  for (int i = 0; i < NUM_LAYERS; i++) {
    assert(addr_mode == net_blocks_cache[i]->addr_mode);
    if (io_alone_reuse) {
      prev_past_key[i] = past_key[i];
      prev_past_value[i] = past_value[i];
    }
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    if (io_alone_reuse) {
      if (i != NUM_LAYERS - 1) {
        assert(prev_past_key[i].u.device.device_addr + prev_past_key[i].size < past_key[i + 1].u.device.device_addr);
        assert(prev_past_value[i].u.device.device_addr + prev_past_value[i].size < past_value[i + 1].u.device.device_addr);

        assert(prev_past_key[i].u.device.device_addr + prev_past_key[i].size < past_value[i + 1].u.device.device_addr);
        assert(prev_past_value[i].u.device.device_addr + prev_past_value[i].size < past_key[i + 1].u.device.device_addr);
      }
      d2d(tmp_key_cache, prev_past_key[i]);
      d2d(tmp_value_cache, prev_past_value[i]);
      d2d(past_key[i], tmp_key_cache);
      d2d(past_value[i], tmp_value_cache);
    }
  }
}

void Qwen::empty_kvcache() {
  int value = 0;
  for (int i = 0; i < NUM_LAYERS; i++) {
    bool status = bm_memset_device_ext(bm_handle, &value, 1, past_key[i]);
    assert(BM_SUCCESS == status);
    status = bm_memset_device_ext(bm_handle, &value, 1, past_value[i]);
    assert(BM_SUCCESS == status);
    status = bm_memset_device_ext(bm_handle, &value, 1, net_blocks_unshare[i]->stages[0].input_mems[3]);
    assert(BM_SUCCESS == status);
    status = bm_memset_device_ext(bm_handle, &value, 1, net_blocks_unshare[i]->stages[0].input_mems[4]);
    assert(BM_SUCCESS == status);
  }
  return;
}

std::vector<uint8_t> Qwen::read_file(std::string model_path, size_t size, size_t offset) {
  std::ifstream file(model_path, std::ios::binary);

  std::vector<uint8_t> data(size);
  file.seekg(offset, std::ios::beg);
  file.read(reinterpret_cast<char*>(data.data()), size);
  file.close();
  return data;
}

std::vector<uint8_t> Qwen::enc_file(std::string model_path, size_t size, size_t offset) {
  auto data = read_file(model_path, size, offset);

  std::vector<uint8_t> encrypted_data(size);
  cipher.encrypt(data, encrypted_data);
  return encrypted_data;
}

std::vector<uint8_t> Qwen::dec_file(std::string model_path, size_t size, size_t offset) {
  auto data = read_file(model_path, size, offset);

  std::vector<uint8_t> decrypted_data(size);
  cipher.decrypt(data, decrypted_data);
  return decrypted_data;
}

// encrypt flatbuffer
void Qwen::encrypt_bmodel(std::string model_path) {
  size_t offset = 0;
  size_t size = 64;

  auto data = read_file(model_path, size, offset);

  // read header
  MODEL_HEADER_T header;
  memcpy(&header, data.data(), sizeof(header));

  // write encrypted_data return to file
  offset = header.header_size;
  size = header.flatbuffers_size;
  auto encrypted_data = enc_file(model_path, size, offset);
  std::fstream outFile(model_path, std::ios::in | std::ios::out | std::ios::binary);
  if (outFile) {
      outFile.seekp(offset, std::ios::beg);
      outFile.write(reinterpret_cast<char*>(encrypted_data.data()), size);
      outFile.close();
  }
}

// decrypt bmodel
std::vector<uint8_t> Qwen::decrypt_bmodel(std::string model_path) {
  size_t header_offset = 0;
  size_t header_size = 64;

  // read header
  auto data = read_file(model_path, header_size, header_offset);

  MODEL_HEADER_T header;
  memcpy(&header, data.data(), sizeof(header));

  // read flatbuffer
  auto dec_offset = header.header_size;
  auto dec_size = header.flatbuffers_size;
  auto decrypted_data = dec_file(model_path, dec_size, dec_offset);
  decrypted_data.insert(decrypted_data.begin(), data.begin(), data.end());

  auto total_size = header.header_size + header.flatbuffers_size + header.binary_size;
  if (bmrt_get_bmodel_info_from_data((void*)decrypted_data.data(), &mem_info, total_size) == false) {
    throw std::runtime_error("Load bmodel Failed");
  }
  return decrypted_data;
}

void Qwen::malloc_bmodel_mem() {

  // if (bmrt_get_bmodel_info(model_path.c_str(), &mem_info) == false) {
  //   throw std::runtime_error("Load bmodel Failed");
  // }
  mem_info.io_mem.size = 0;

  // free all memory without coeff memory
  if (prealloc_mem_v.size() == 0) {
    malloc_device_mem(bm_handle, mem_info.coeff_mem, prealloc_mem_v);
  } else {
    mem_info.coeff_mem.addr = prealloc_mem_v[0].u.device.device_addr;
  }
  malloc_device_mem(bm_handle, mem_info.instruction_mem, prealloc_mem_v);
  malloc_device_mem(bm_handle, mem_info.variable_instruction_mem, prealloc_mem_v);
  malloc_device_mem(bm_handle, mem_info.neuron_mem, prealloc_mem_v);
  malloc_device_mem(bm_handle, mem_info.io_mem, prealloc_mem_v);
}

void Qwen::free_device() {
  while (prealloc_mem_v.size() > 1) {
    bm_free_device_u64(bm_handle, prealloc_mem_v.back());
    prealloc_mem_v.pop_back();
  }
}

void Qwen::deinit() {
  for (size_t i = 0; i < prealloc_mem_v.size(); ++i) {
    bm_free_device_u64(bm_handle, prealloc_mem_v[i]);
  }
  // for (size_t i = 0; i < io_mem_v.size(); i++) {
  //   bm_free_device(bm_handle, io_mem_v[i]);
  // }
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

int Qwen::penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem, 
                         std::vector<int> &input_tokens, int &token_length) {
  auto &in1_mem = net->stages[0].input_mems[1];
  auto &in2_mem = net->stages[0].input_mems[2];
  auto &in3_mem = net->stages[0].input_mems[3];
  auto &in4_mem = net->stages[0].input_mems[4];
  auto &out0_mem = net->stages[0].output_mems[0];
  auto &out1_mem = net->stages[0].output_mems[1];

  // repeat_penalty + top_p + top_k + temperature
  std::vector<int> generated_tokens(SEQLEN, input_tokens[token_length - 1]);
  repeat_last_n = std::min(repeat_last_n, token_length);
  std::copy(input_tokens.begin() + token_length - repeat_last_n, 
            input_tokens.begin() + token_length,
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

void Qwen::forward_first(std::vector<int> &tokens) {
  std::vector<int> share_tokens(MAX_SHARE_LENGTH, 0);
  std::vector<int> position_id(MAX_SHARE_LENGTH, 0);
  std::vector<uint16_t> attention_mask(MAX_SHARE_LENGTH * MAX_SHARE_LENGTH, mask_value);
  std::copy(tokens.begin(), tokens.end(), share_tokens.data());
  
  share_length = tokens.size();

  for (int i = 0; i < share_length; i++) {
    position_id[i] = i;
  }
  for (int i = 0; i < share_length; i++) {
    for (int j = 0; j < MAX_SHARE_LENGTH; j++) {
      if (j <= i) {
        attention_mask[i * MAX_SHARE_LENGTH + j] = 0;
      }
    }
  }

  // forward embeding
  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)share_tokens.data());
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
    if (is_dynamic) {
      dynamic_net_launch(net_blocks[idx], share_length);
    } else {
      net_launch(net_blocks[idx]);
    }
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1], 0);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2], 0);
  }
  return;
}

int Qwen::forward_unshare(std::vector<int> &tokens) {
  std::vector<int> position_id(MAX_UNSHARE_LENGTH, 0);
  std::vector<uint16_t> attention_mask(MAX_UNSHARE_LENGTH * (MAX_SHARE_LENGTH + MAX_UNSHARE_LENGTH), mask_value);
  unshare_tokens.clear();
  unshare_tokens.resize(SEQLEN - MAX_SHARE_LENGTH);
  std::copy(tokens.begin(), tokens.end(), unshare_tokens.data());
  
  unshare_length = tokens.size();

  for (int i = 0; i < unshare_length; i++) {
    position_id[i] = i + share_length;
  }
  for (int i = 0; i < unshare_length; i++) {
    for (int j = 0; j < share_length; j++) {
      attention_mask[i * (MAX_SHARE_LENGTH + MAX_UNSHARE_LENGTH) + j] = 0;
    }
    for (int j = MAX_SHARE_LENGTH; j < MAX_SHARE_LENGTH + MAX_UNSHARE_LENGTH; j++) {
      if (j - MAX_SHARE_LENGTH <= i) {
        attention_mask[i * (MAX_SHARE_LENGTH + MAX_UNSHARE_LENGTH) + j] = 0;
      }
    }
  }

  // forward embeding
  auto &in_mem = net_embed_unshare->stages[0].input_mems[0];
  auto &out_mem = net_embed_unshare->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)unshare_tokens.data());
  net_launch(net_embed_unshare); // prefil embedding

  // forward blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_unshare[0]->stages[0].input_mems[3]) / MAX_SHARE_LENGTH;
  // int share_size = share_length * bytes;
  int max_share_offset = share_length * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks_unshare[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks_unshare[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks_unshare[idx]->stages[0].input_mems[2];
    auto &in3_mem = net_blocks_unshare[idx]->stages[0].input_mems[3];
    auto &in4_mem = net_blocks_unshare[idx]->stages[0].input_mems[4];
    d2d(in0_mem, out_mem);
    if (io_alone) {
      if (idx == 0) {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      } else {
        d2d(in1_mem, net_blocks_unshare[0]->stages[0].input_mems[1]);
        d2d(in2_mem, net_blocks_unshare[0]->stages[0].input_mems[2]);
      }
      d2d(in3_mem, past_key[idx]);
      d2d(in4_mem, past_value[idx]);
    } else {
      throw std::runtime_error("Only support io_alone");
    }
    net_launch(net_blocks_unshare[idx]);
    out_mem = net_blocks_unshare[idx]->stages[0].output_mems[0];
    d2d(past_key[idx], net_blocks_unshare[idx]->stages[0].output_mems[1], max_share_offset);
    d2d(past_value[idx], net_blocks_unshare[idx]->stages[0].output_mems[2], max_share_offset);
  }

  // forward lmhead
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (unshare_length - 1) * bytes, bytes);
  net_launch(net_lm);

  int token = 0;
  if (generation_mode == "greedy") {
    token = greedy_search(net_greedy_head, lm_out_mem);
  } else if (generation_mode == "penalty_sample") {
    token = penalty_sample(net_penalty_sample_head, lm_out_mem, unshare_tokens, unshare_length);
  }

  unshare_tokens[unshare_length] = token;
  unshare_length += 1;
  return token;
}

int Qwen::forward_next() {
  int cur_token = unshare_tokens[unshare_length - 1];

  // std::vector<uint16_t> attention_mask(SEQLEN + 1, mask_value);
  // for (int i = 0; i < share_length; i++) {
  //   attention_mask[i] = 0;
  // }
  // for (int i = MAX_SHARE_LENGTH; i < MAX_SHARE_LENGTH + unshare_length - 1; i++) {
  //   attention_mask[i] = 0;
  // }
  // attention_mask[SEQLEN] = 0;

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = share_length + unshare_length; i < SEQLEN; i++) {
    attention_mask[i] = mask_value;
  }
  int32_t position_id = share_length + unshare_length - 1;

  // embedding
  auto &in_mem = net_embed_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)&cur_token);
  net_launch(net_embed_cache);

  // blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (share_length + unshare_length - 1) * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks_cache[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks_cache[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks_cache[idx]->stages[0].input_mems[2];
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
      throw std::runtime_error("Only support io_alone");
    }
    net_launch(net_blocks_cache[idx]);
    out_mem = out0_mem;
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], token_offset, out1_mem, 0,
                       bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], token_offset, out2_mem, 0,
                       bytes);
    // dump_tensor_to_file<uint16_t>(bm_handle,net_blocks_cache[idx]->stages[0].output_mems[0],{1,1,2560},"output_" + std::to_string(idx) + ".npz","hidden_states");
    // dump_tensor_to_file<uint16_t>(bm_handle,net_blocks_cache[idx]->stages[0].output_mems[1],{1, 1, 20, 128},"output_" + std::to_string(idx) + ".npz","present_key");
    // dump_tensor_to_file<uint16_t>(bm_handle,net_blocks_cache[idx]->stages[0].output_mems[2],{1, 1, 20, 128},"output_" + std::to_string(idx) + ".npz","present_value");
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
    token = penalty_sample(net_penalty_sample_head, lm_out_mem, unshare_tokens, unshare_length);
  }
  
  unshare_tokens[unshare_length] = token;
  unshare_length += 1;
  return token;
}


// std::vector<int> Qwen::generate(std::vector<int> &history_tokens, int EOS) {
//   if (history_tokens.empty()) {
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
        .def("encrypt_bmodel", &Qwen::encrypt_bmodel)
        .def("forward_first", &Qwen::forward_first)
        .def("forward_unshare", &Qwen::forward_unshare)
        .def("forward_next", &Qwen::forward_next)
        .def("free_device", &Qwen::free_device)
        .def("deinit", &Qwen::deinit)
        .def("empty_kvcache", &Qwen::empty_kvcache)
        .def_readwrite("SEQLEN", &Qwen::SEQLEN) // read SEQLEN in pipeline.py
        .def_readwrite("MAX_SHARE_LENGTH", &Qwen::MAX_SHARE_LENGTH)
        .def_readwrite("share_length", &Qwen::share_length)
        .def_readwrite("unshare_length", &Qwen::unshare_length)
        .def_readwrite("unshare_tokens", &Qwen::unshare_tokens)
        .def_readwrite("temperature", &Qwen::temperature)
        .def_readwrite("top_p", &Qwen::top_p)
        .def_readwrite("repeat_penalty", &Qwen::repeat_penalty)
        .def_readwrite("repeat_last_n", &Qwen::repeat_last_n)
        .def_readwrite("max_new_tokens", &Qwen::max_new_tokens)
        .def_readwrite("generation_mode", &Qwen::generation_mode)
        .def_readwrite("prompt_mode", &Qwen::prompt_mode)
        .def_readwrite("memory_prealloc", &Qwen::memory_prealloc)
        .def_readwrite("io_alone_reuse", &Qwen::io_alone_reuse)
        .def_readwrite("is_decrypt", &Qwen::is_decrypt);
}