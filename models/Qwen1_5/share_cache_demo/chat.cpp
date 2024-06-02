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

#include "bmruntime_interface.h"
#include "memory.h"
#include "utils.h"

static const float ATTENTION_MASK = -10000.;

// template <typename T>
// void dump_tensor_to_file(
//         bm_handle_t&          handle,
//         bm_device_mem_t&          t,
//         std::vector<size_t>&& shape,
//         const std::string&    filename,
//         const std::string&    tensor_name) {
//     int  cnt = bm_mem_get_device_size(t) / sizeof(T);
//     auto buffer = std::make_unique<T[]>(cnt);
//     bm_memcpy_d2s(handle, buffer.get(), t);
    
//     if constexpr (std::is_same_v<T, uint16_t>) {
//       std::vector<float> data(cnt);
//       for (int i = 0; i < cnt; i++)
//         data[i] = fp16_ieee_to_fp32_value(buffer[i]);
//       cnpy::npz_save(filename, tensor_name, data.data(), shape, "a");
//     } else if constexpr (std::is_same_v<T, int32_t>){
//       std::vector<int> data(cnt);
//       memcpy(data.data(), buffer.get(), sizeof(int) * cnt);
//       cnpy::npz_save(filename, tensor_name, data.data(), shape, "a");
//     } else {
//       std::vector<float> data(cnt);
//       memcpy(data.data(), buffer.get(), sizeof(float) * cnt);
//       cnpy::npz_save(filename, tensor_name, data.data(), shape, "a");
//     }
// }

class Qwen {
public:
  void init(const std::vector<int> &devid, std::string model_path);
  void deinit();
  void forward_first(std::vector<int> &tokens);
  int forward_unshare(std::vector<int> &tokens);
  int forward_next();
  std::vector<int> generate(std::vector<int> &history_tokens, int EOS);

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
  std::vector<int> unshare_tokens;
  uint16_t mask_value;

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
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
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

  is_dynamic = net_blocks[0]->is_dynamic;
  auto addr_mode = net_blocks_cache[0]->addr_mode;
  io_alone = addr_mode == 1;
  for (int i = 0; i < NUM_LAYERS; i++) {
    assert(addr_mode == net_blocks_cache[i]->addr_mode);
    if (io_alone) {
      past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
      past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    } else {
      throw std::runtime_error("Only support io_alone");
    }
  }

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
  std::vector<int> visited_tokens(MAX_SHARE_LENGTH, 0);
  std::vector<int> position_id(MAX_SHARE_LENGTH, 0);
  std::vector<uint16_t> attention_mask(MAX_SHARE_LENGTH * MAX_SHARE_LENGTH, mask_value);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());
  
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

    dynamic_net_launch(net_blocks[idx], share_length);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1], 0);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2], 0);
  }
  return;
}

int Qwen::forward_unshare(std::vector<int> &tokens) {
  std::vector<int> visited_tokens(MAX_UNSHARE_LENGTH, 0);
  std::vector<int> position_id(MAX_UNSHARE_LENGTH, 0);
  std::vector<uint16_t> attention_mask(MAX_UNSHARE_LENGTH * (MAX_SHARE_LENGTH + MAX_UNSHARE_LENGTH), mask_value);
  unshare_tokens.clear();
  unshare_tokens.resize(SEQLEN - MAX_SHARE_LENGTH);
  std::copy(tokens.begin(), tokens.end(), unshare_tokens.data());
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());
  
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
  bm_memcpy_s2d(bm_handle, in_mem, (void *)visited_tokens.data());
  net_launch(net_embed_unshare); // prefil embedding

  // forward blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_unshare[0]->stages[0].input_mems[3]) / MAX_SHARE_LENGTH;
  // int share_size = share_length * bytes;
  int max_share_offset = MAX_SHARE_LENGTH * bytes;
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

  std::vector<uint16_t> attention_mask(SEQLEN + 1, mask_value);
  for (int i = 0; i < share_length; i++) {
    attention_mask[i] = 0;
  }
  for (int i = MAX_SHARE_LENGTH; i < MAX_SHARE_LENGTH + unshare_length - 1; i++) {
    attention_mask[i] = 0;
  }
  attention_mask[SEQLEN] = 0;
  int32_t position_id = share_length + unshare_length - 1;

  // embedding
  auto &in_mem = net_embed_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)&cur_token);
  net_launch(net_embed_cache);

  // blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (MAX_SHARE_LENGTH + unshare_length - 1) * bytes;
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
        .def("forward_first", &Qwen::forward_first)
        .def("forward_unshare", &Qwen::forward_unshare)
        .def("forward_next", &Qwen::forward_next)
        .def("deinit", &Qwen::deinit)
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
        .def_readwrite("prompt_mode", &Qwen::prompt_mode);
}

