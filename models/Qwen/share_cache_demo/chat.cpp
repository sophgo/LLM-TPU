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
#include "cnpy.h"

static const float ATTENTION_MASK = -10000.;

template <typename T>
void dump_tensor_to_file(
        bm_handle_t&          handle,
        bm_device_mem_t&          t,
        std::vector<size_t>&& shape,
        const std::string&    filename,
        const std::string&    tensor_name) {
    int  cnt = bm_mem_get_device_size(t) / sizeof(T);
    auto buffer = std::make_unique<T[]>(cnt);
    bm_memcpy_d2s(handle, buffer.get(), t);
    std::vector<T> data(cnt);
    if constexpr (std::is_same_v<T, unsigned short>) {
        for (int i = 0; i < cnt; i++)
            data[i] = half_to_float(buffer[i]);
    } else {
        memcpy(data.data(), buffer.get(), sizeof(float) * cnt);
    }
    cnpy::npz_save(filename, tensor_name, data.data(), shape, "a");
}

class Qwen {
public:
  void init(const std::vector<int> &devid, std::string model_path);
  void deinit();
  void forward_first(std::vector<int> &tokens);
  void forward_share_first(std::vector<int> &tokens, int batch_id);
  void forward_share_next();
  std::vector<int> generate(std::vector<int> &history_tokens, int EOS);

  std::mt19937 sgen;
  Qwen() : sgen(std::random_device()()){};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset);

  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem, std::vector<int> &visited_tokens, int &token_length);

public:
  int share_length;
  std::vector<int> unshare_length;
  int SEQLEN;     // read from bmodel
  int NUM_LAYERS; // read from bmodel
  int MAX_SHARE_LENGTH;
  int MAX_UNSHARE_LENGTH;
  int BATCH_SIZE;
  bool io_alone;
  std::vector<int> visited_tokens;
  std::vector<int> share_tokens;
  std::vector<std::vector<int>> unshare_tokens;
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
  std::vector<const bm_net_info_t *> net_blocks_share;
  std::vector<const bm_net_info_t *> net_blocks_share_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_unshare;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm, *net_greedy_head, *net_penalty_sample_head;
  std::vector<bm_device_mem_t> share_past_key;
  std::vector<bm_device_mem_t> share_past_value;
  std::vector<bm_device_mem_t> unshare_past_key;
  std::vector<bm_device_mem_t> unshare_past_value;
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
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(dst));
}

void Qwen::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset) {
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, bm_mem_get_device_size(src));
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
  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1]; // real seqlen
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = (num_nets - 5) / 3;

  // resize
  unshare_tokens.resize(BATCH_SIZE);
  unshare_length.resize(BATCH_SIZE + 10);

  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto block_share_name = "block_share_" + std::to_string(i);
    auto block_share_cache_name = "block_share_cache_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_share.emplace_back(
        bmrt_get_network_info(p_bmrt, block_share_name.c_str()));
    net_blocks_share_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, block_share_cache_name.c_str()));
  }

  MAX_UNSHARE_LENGTH = net_blocks_share[0]->stages[0].input_shapes[0].dims[1]; // real seqlen
  MAX_SHARE_LENGTH = net_blocks_share[0]->stages[0].input_shapes[3].dims[1]; // real seqlen
  BATCH_SIZE = net_blocks_share_cache[0]->stages[0].input_shapes[0].dims[0]; // real seqlen

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
  share_past_key.resize(NUM_LAYERS);
  share_past_value.resize(NUM_LAYERS);
  unshare_past_key.resize(NUM_LAYERS);
  unshare_past_value.resize(NUM_LAYERS);
  auto addr_mode = net_blocks_share_cache[0]->addr_mode;
  io_alone = addr_mode == 1;
  for (int i = 0; i < NUM_LAYERS; i++) {
    assert(addr_mode == net_blocks_share_cache[i]->addr_mode);
    if (io_alone) {
      share_past_key[i] = net_blocks_share_cache[i]->stages[0].input_mems[4];
      share_past_value[i] = net_blocks_share_cache[i]->stages[0].input_mems[5];
      unshare_past_key[i] = net_blocks_share_cache[i]->stages[0].input_mems[6];
      unshare_past_value[i] = net_blocks_share_cache[i]->stages[0].input_mems[7];
    } else {
      throw std::runtime_error("Only support io_alone");
    }
  }
}

void Qwen::deinit() {
  if (false == io_alone) {
    for (int i = 0; i < NUM_LAYERS; i++) {
      bm_free_device(bm_handle, share_past_key[i]);
      bm_free_device(bm_handle, share_past_value[i]);
      bm_free_device(bm_handle, unshare_past_key[i]);
      bm_free_device(bm_handle, unshare_past_value[i]);
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
  std::vector<int> generated_tokens(MAX_SHARE_LENGTH + MAX_UNSHARE_LENGTH, input_tokens[token_length - 1]);
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
  std::vector<int> position_id(MAX_SHARE_LENGTH, 0);
  std::vector<uint16_t> attention_mask(MAX_SHARE_LENGTH * MAX_SHARE_LENGTH, mask_value);
  
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
  bm_memcpy_s2d(bm_handle, in_mem, (void *)tokens.data());
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
    d2d(share_past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
    d2d(share_past_value[idx], net_blocks[idx]->stages[0].output_mems[2]);
  }
  return;
}


void Qwen::forward_share_first(std::vector<int> &tokens, int batch_id) {
  unshare_tokens[batch_id].resize(MAX_UNSHARE_LENGTH);
  std::vector<int> position_id(MAX_UNSHARE_LENGTH, 0);
  std::vector<uint16_t> attention_mask(MAX_UNSHARE_LENGTH * (MAX_SHARE_LENGTH + MAX_UNSHARE_LENGTH), mask_value);
  std::copy(tokens.begin(), tokens.end(), unshare_tokens[batch_id].data());
  
  unshare_length[batch_id] = tokens.size();

  for (int i = 0; i < unshare_length[batch_id]; i++) {
    position_id[i] = i + share_length;
  }
  for (int i = 0; i < share_length + unshare_length[batch_id]; i++) {
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
  bm_memcpy_s2d(bm_handle, in_mem, (void *)unshare_tokens[batch_id].data());
  net_launch(net_embed_unshare); // prefil embedding

  // forward blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_share[0]->stages[0].input_mems[3]) / MAX_SHARE_LENGTH;
  int unshare_offset = MAX_UNSHARE_LENGTH * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks_share[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks_share[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks_share[idx]->stages[0].input_mems[2];
    auto &in3_mem = net_blocks_share[idx]->stages[0].input_mems[3];
    auto &in4_mem = net_blocks_share[idx]->stages[0].input_mems[4];
    d2d(in0_mem, out_mem);
    if (io_alone) {
      if (idx == 0) {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      } else {
        d2d(in1_mem, net_blocks_share[0]->stages[0].input_mems[1]);
        d2d(in2_mem, net_blocks_share[0]->stages[0].input_mems[2]);
      }
      d2d(in3_mem, share_past_key[idx]);
      d2d(in4_mem, share_past_value[idx]);
    } else {
      throw std::runtime_error("Only support io_alone");
    }
    net_launch(net_blocks_share[idx]);
    out_mem = net_blocks_share[idx]->stages[0].output_mems[0];
    d2d(unshare_past_key[idx], net_blocks_share[idx]->stages[0].output_mems[1], unshare_offset * batch_id);
    d2d(unshare_past_value[idx], net_blocks_share[idx]->stages[0].output_mems[2], unshare_offset * batch_id);
  }

  // forward lmhead
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (unshare_length[batch_id] - 1) * bytes, bytes);
  net_launch(net_lm);

  int token = 0;
  if (generation_mode == "greedy") {
    token = greedy_search(net_greedy_head, lm_out_mem);
  } else if (generation_mode == "penalty_sample") {
    token = penalty_sample(net_penalty_sample_head, lm_out_mem, unshare_tokens[batch_id], unshare_length[batch_id]);
  }

  unshare_tokens[batch_id][unshare_length[batch_id]] = token;
  unshare_length[batch_id] += 1;
  visited_tokens.push_back(token);
  return;
}


void Qwen::forward_share_next() {
  std::vector<int> cur_token;
  for (int i = 0; i < BATCH_SIZE; i++) {
    cur_token.push_back(unshare_tokens[i][unshare_length[i] - 1]);
  }

  std::vector<uint16_t> share_attention_mask(MAX_SHARE_LENGTH, 0);
  std::vector<uint16_t> unshare_attention_mask(BATCH_SIZE * MAX_UNSHARE_LENGTH, 0);
  for (int i = share_length; i < MAX_SHARE_LENGTH; i++) {
    share_attention_mask[i] = mask_value;
  }
  for (int i = 0; i < BATCH_SIZE; i++) {
    for (int j = unshare_length[i] - 1; j < MAX_UNSHARE_LENGTH; j++) {
      unshare_attention_mask[j + i * MAX_UNSHARE_LENGTH] = mask_value;
    }
  }

  std::vector<int32_t> position_id;
  for (int i = 0; i < BATCH_SIZE; i++) {
    position_id.push_back(unshare_length[i] - 1 + share_length);
  }

  // embedding
  auto &in_mem = net_embed_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)cur_token.data());
  net_launch(net_embed_cache);

  // blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_share_cache[0]->stages[0].output_mems[1]) / BATCH_SIZE;
  std::vector<int> token_offset;
  for (int i = 0; i < BATCH_SIZE; i++) {
    token_offset.push_back((unshare_length[i] - 1) * bytes);
  }
  int batch_offset = MAX_UNSHARE_LENGTH * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks_share_cache[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks_share_cache[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks_share_cache[idx]->stages[0].input_mems[2];
    auto &in3_mem = net_blocks_share_cache[idx]->stages[0].input_mems[3];
    auto &in4_mem = net_blocks_share_cache[idx]->stages[0].input_mems[4];
    auto &in5_mem = net_blocks_share_cache[idx]->stages[0].input_mems[5];
    auto &in6_mem = net_blocks_share_cache[idx]->stages[0].input_mems[6];
    auto &in7_mem = net_blocks_share_cache[idx]->stages[0].input_mems[7];
    auto &out0_mem = net_blocks_share_cache[idx]->stages[0].output_mems[0];
    auto &out1_mem = net_blocks_share_cache[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks_share_cache[idx]->stages[0].output_mems[2];
    d2d(in0_mem, out_mem);
    if (io_alone) {
      if (idx == 0) {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)share_attention_mask.data());
        bm_memcpy_s2d(bm_handle, in3_mem, (void *)unshare_attention_mask.data());
      } else {
        d2d(in1_mem, net_blocks_share_cache[0]->stages[0].input_mems[1]);
        d2d(in2_mem, net_blocks_share_cache[0]->stages[0].input_mems[2]);
        d2d(in3_mem, net_blocks_share_cache[0]->stages[0].input_mems[3]);
      }
    }  else {
      d2d(in4_mem, share_past_key[idx]);
      d2d(in5_mem, share_past_value[idx]);
      d2d(in6_mem, unshare_past_key[idx]);
      d2d(in7_mem, unshare_past_value[idx]);
    }

    // dump_tensor_to_file<float>(bm_handle,in0_mem,{2,1,4096},"inputs.npz","input_states");
    // dump_tensor_to_file<int32_t>(bm_handle,in1_mem,{2,1},"inputs.npz","position_ids");
    // dump_tensor_to_file<float>(bm_handle,in2_mem,{1,1,1,6144},"inputs.npz","share_attention_mask");
    // dump_tensor_to_file<float>(bm_handle,in3_mem,{2,1,1,1024},"inputs.npz","unshare_attention_mask");
    // dump_tensor_to_file<float>(bm_handle,in4_mem,{1,6144,32,128},"inputs.npz","share_past_k");
    // dump_tensor_to_file<float>(bm_handle,in5_mem,{1,6144,32,128},"inputs.npz","share_past_v");
    // dump_tensor_to_file<float>(bm_handle,in6_mem,{2,1024,32,128},"inputs.npz","unshare_past_k");
    // dump_tensor_to_file<float>(bm_handle,in7_mem,{2,1024,32,128},"inputs.npz","unshare_past_v");
    net_launch(net_blocks_share_cache[idx]);
    out_mem = out0_mem;
    for (int batch_id = 0; batch_id < BATCH_SIZE; batch_id++) {
      bm_memcpy_d2d_byte(bm_handle, unshare_past_key[idx], batch_offset * batch_id + token_offset[batch_id], out1_mem, bytes * batch_id,
                         bytes);
      bm_memcpy_d2d_byte(bm_handle, unshare_past_value[idx], batch_offset * batch_id + token_offset[batch_id], out2_mem, bytes * batch_id,
                         bytes);
    }
  }

  // forward lmhead
  for (int batch_id = 0; batch_id < BATCH_SIZE; batch_id++) {
    auto &lm_in_mem = net_lm->stages[0].input_mems[0];
    auto &lm_out_mem = net_lm->stages[0].output_mems[0];
    bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem, bytes * batch_id,
                       bytes);
    net_launch(net_lm);

    int token = 0;
    if (generation_mode == "greedy") {
      token = greedy_search(net_greedy_head, lm_out_mem);
    } else if (generation_mode == "penalty_sample") {
      token = penalty_sample(net_penalty_sample_head, lm_out_mem, unshare_tokens[batch_id], unshare_length[batch_id]);
    }
    unshare_tokens[batch_id][unshare_length[batch_id]] = token;
    unshare_length[batch_id] += 1;
    visited_tokens.push_back(token);
  }
  return;
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
        .def("forward_share_first", &Qwen::forward_share_first)
        .def("forward_share_next", &Qwen::forward_share_next)
        .def("deinit", &Qwen::deinit)
        .def_readwrite("SEQLEN", &Qwen::SEQLEN) // read SEQLEN in pipeline.py
        .def_readwrite("share_length", &Qwen::share_length)
        .def_readwrite("visited_tokens", &Qwen::visited_tokens)
        .def_readwrite("temperature", &Qwen::temperature)
        .def_readwrite("top_p", &Qwen::top_p)
        .def_readwrite("repeat_penalty", &Qwen::repeat_penalty)
        .def_readwrite("repeat_last_n", &Qwen::repeat_last_n)
        .def_readwrite("max_new_tokens", &Qwen::max_new_tokens)
        .def_readwrite("generation_mode", &Qwen::generation_mode)
        .def_readwrite("prompt_mode", &Qwen::prompt_mode);
}
