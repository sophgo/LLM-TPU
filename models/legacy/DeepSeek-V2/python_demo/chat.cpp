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
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <getopt.h>
#include <inttypes.h>
#include <iostream>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdio.h>
#include <vector>

static void print_devmem_info(bm_handle_t &bm_handle) {
  bm_dev_stat_t stat;
  auto ret = bm_get_stat(bm_handle, &stat);
  if (ret != BM_SUCCESS) {
    std::cerr << "Failed to get device status" << std::endl;
    return;
  }
  std::cout << "DevMem: " << stat.mem_used << "/" << stat.mem_total << " MB"
            << std::endl;
}

static const uint16_t ATTENTION_MASK = 0xC61C;
//===------------------------------------------------------------===//
// Empty Func
//===------------------------------------------------------------===//
void empty(bm_handle_t &bm_handle, bm_device_mem_t &mem) {
  int value = 0;
  auto ret = bm_memset_device_ext(bm_handle, &value, 1, mem);
  assert(BM_SUCCESS == ret);
}

void empty_net(bm_handle_t &bm_handle, const bm_net_info_t *net,
               int stage_idx = 0) {
  for (int i = 0; i < net->input_num; i++) {
    empty(bm_handle, net->stages[stage_idx].input_mems[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    empty(bm_handle, net->stages[stage_idx].output_mems[i]);
  }
}

class Model {
public:
  void init(const std::vector<int> &devid, std::string model_path);
  void deinit();
  int forward_first(std::vector<int> &tokens);
  int forward_next();
  std::vector<int> generate(std::vector<int> &history_tokens, int EOS);

  std::mt19937 sgen;
  Model() : sgen(std::random_device()()){};

private:
  // The following helper functions are unchanged
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  void net_launch_dyn(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
                  int size);

  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem);

public:
  int hidden_bytes;
  int kv_bytes;
  int token_length;
  int SEQLEN;     // The real seqlen read from the bmodel
  int NUM_LAYERS; // Total number of layers
  int TOKEN_LEN;
  bool is_dynamic;
  std::vector<int> visited_tokens;

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

  // Modules of the model:
  // The first layer uses the attention / mlp modules
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;

  std::vector<const bm_net_info_t *> attention; // layer0 uses attention_0
  std::vector<const bm_net_info_t *>
      attention_cache; // The cache of layer0 corresponds to attention_cache_0
  std::vector<const bm_net_info_t *> mlp; // layer0 uses mlp_0
  std::vector<const bm_net_info_t *>
      mlp_cache; // The cache of layer0 corresponds to mlp_cache_0

  // The second layer and beyond use the shared moe structure
  std::vector<const bm_net_info_t *> shared_moe;
  std::vector<const bm_net_info_t *> shared_moe_cache;
  std::vector<const bm_net_info_t *> moe;
  std::vector<const bm_net_info_t *> moe_cache;

  // lm_head and generation heads
  const bm_net_info_t *net_lm;
  const bm_net_info_t *net_greedy_head;
  const bm_net_info_t *net_penalty_sample_head;

  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};

void Model::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void Model::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset) {
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0,
                     bm_mem_get_device_size(src));
}

void Model::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
                int size) {
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void Model::init(const std::vector<int> &devices, std::string model_path) {

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
  print_devmem_info(handles[0]);

  // Get the embedding and lm modules
  // net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  // net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  // net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  // net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
  // net_penalty_sample_head = bmrt_get_network_info(p_bmrt,
  // "penalty_sample_head");

  // Define the real SEQLEN based on the embedding layer
  SEQLEN = 512; // real seqlen

  // Compute NUM_LAYERS, determined by the number of nets in the bmodel excluding the fixed modules
  // Note: the formula for the total number of layers must ensure that layer0 uses the mlp module and all other layers use the shared moe structure
  // auto num_nets = bmrt_get_network_number(p_bmrt);
  // Assume that in the bmodel layer0 has 2 modules (attention_0, mlp_0) plus their caches,
  // and each remaining layer has 4 modules (attention, shared_moe, moe, and the corresponding caches); additionally subtract
  // the embedding, lmhead, and head modules.
  // The formula below is used here as an example only; it may need adjustment in practice:
  NUM_LAYERS = 2;

  // resize visited_tokens
  visited_tokens.resize(SEQLEN);

  // Handle layer0: attention_0 and mlp_0
  {
    // attention module
    auto attn = bmrt_get_network_info(p_bmrt, "attention_0");
    attention.push_back(attn);
    // attention cache module
    auto attn_cache = bmrt_get_network_info(p_bmrt, "attention_cache_0");
    attention_cache.push_back(attn_cache);
    // mlp module
    auto mlp_net = bmrt_get_network_info(p_bmrt, "mlp_0");
    mlp.push_back(mlp_net);
    // mlp cache module
    auto mlp_cache_net = bmrt_get_network_info(p_bmrt, "mlp_cache_0");
    mlp_cache.push_back(mlp_cache_net);
  }
  // Handle layers 1 to NUM_LAYERS-1: shared moe and moe modules
  for (int i = 1; i < NUM_LAYERS; i++) {
    // attention module
    auto attn = bmrt_get_network_info(
        p_bmrt, ("attention_" + std::to_string(i)).c_str());
    attention.push_back(attn);
    // attention cache module
    auto attn_cache = bmrt_get_network_info(
        p_bmrt, ("attention_cache_" + std::to_string(i)).c_str());
    attention_cache.push_back(attn_cache);
    // shared moe module
    auto s_moe = bmrt_get_network_info(
        p_bmrt, ("shared_moe_" + std::to_string(i)).c_str());
    shared_moe.push_back(s_moe);
    // shared moe cache module
    auto s_moe_cache = bmrt_get_network_info(
        p_bmrt, ("shared_moe_cache_" + std::to_string(i)).c_str());
    shared_moe_cache.push_back(s_moe_cache);
    // moe module
    auto moe_net =
        bmrt_get_network_info(p_bmrt, ("moe_" + std::to_string(i)).c_str());
    moe.push_back(moe_net);
    // moe cache module
    auto moe_cache_net = bmrt_get_network_info(
        p_bmrt, ("moe_cache_" + std::to_string(i) + "_0").c_str());
    moe_cache.push_back(moe_cache_net);
  }

  // Device memory sizes (taking layer0 mlp_cache as an example)
  hidden_bytes = bm_mem_get_device_size(mlp_cache[0]->stages[0].output_mems[0]);
  kv_bytes = bm_mem_get_device_size(mlp_cache[0]->stages[0].output_mems[1]);
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
}

void Model::deinit() {
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
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
  assert(ret);
  // bm_thread_sync(bm_handle);
}

void Model::net_launch_dyn(const bm_net_info_t *net, int stage_idx) {
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

  in_tensors[0].shape.dims[1] = TOKEN_LEN;
  in_tensors[1].shape.dims[1] = TOKEN_LEN;
  in_tensors[2].shape.dims[2] = TOKEN_LEN;
  in_tensors[2].shape.dims[3] = TOKEN_LEN;

  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  // bm_thread_sync(bm_handle);
}

void Model::head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
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

int Model::greedy_search(const bm_net_info_t *net,
                         bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];
  head_launch(net, logits_mem);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

int Model::penalty_sample(const bm_net_info_t *net,
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

int Model::forward_first(std::vector<int> &tokens) {
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());

  token_length = tokens.size();
  TOKEN_LEN = tokens.size();

  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  if (is_dynamic) {
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j < TOKEN_LEN; j++) {
        if (j <= i) {
          attention_mask[i * TOKEN_LEN + j] = 0;
        }
      }
    }
  } else {
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j < SEQLEN; j++) {
        if (j <= i) {
          attention_mask[i * SEQLEN + j] = 0;
        }
      }
    }
  }

  // auto start0 = std::chrono::high_resolution_clock::now();
  // for (int i = 0; i < 60; i++) {
  //   net_launch(attention_cache[0]);
  //   net_launch(mlp_cache[0]);
  //   net_launch(shared_moe_cache[0]);
  //   for(int i = 0; i < 6; i++) {
  //     net_launch(moe_cache[0]);
  //   }
  // }
  // auto end0 = std::chrono::high_resolution_clock::now();
  // auto duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0
  // - start0); std::cout << "net_launch execution time: " << duration0.count()
  // << " ms" << std::endl;

  // auto start = std::chrono::high_resolution_clock::now();
  // for (int i = 0; i < 60; i++) {
  //   net_launch(attention[0]);
  //   net_launch(mlp[0]);
  //   net_launch(shared_moe[0]);
  //   net_launch(moe[0]);
  // }
  // auto end = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end -
  // start); std::cout << "net_launch execution time: " << duration.count() << "
  // ms" << std::endl;

  // Clear the inputs of each module (calling the empty function here, assuming its definition is the same as before)
  // For the layer0 modules:
  empty_net(bm_handle, attention[0]);
  empty_net(bm_handle, mlp[0]);
  empty_net(bm_handle, attention_cache[0]);
  empty_net(bm_handle, mlp_cache[0]);
  // For layer1 and beyond:
  for (int idx = 1; idx < NUM_LAYERS; idx++) {
    empty_net(bm_handle, attention[idx]);
    empty_net(bm_handle, shared_moe[idx - 1]);
    empty_net(bm_handle, moe[idx - 1]);
    empty_net(bm_handle, attention_cache[idx]);
    empty_net(bm_handle, shared_moe_cache[idx - 1]);
    empty_net(bm_handle, moe_cache[idx - 1]);
  }

  // forward embedding
  auto in_mem = net_embed->stages[0].input_mems[0];
  auto out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)visited_tokens.data());
  net_launch(net_embed); // run embedding

  // First layer: layer0
  // forward attention_0: here we assume the embedding layer output is copied directly to the attention module
  auto &in0_mem = attention[0]->stages[0].input_mems[0];
  empty(bm_handle, attention[0]->stages[0].input_mems[0]);
  d2d(in0_mem, out_mem, 0, token_length * hidden_bytes);
  // For layer0 attention, position and attention_mask are only passed in the first time
  bm_memcpy_s2d(bm_handle, attention[0]->stages[0].input_mems[1],
                (void *)position_id.data());
  bm_memcpy_s2d(bm_handle, attention[0]->stages[0].input_mems[2],
                (void *)attention_mask.data());
  if (is_dynamic)
    net_launch_dyn(attention[0]);
  else
    net_launch(attention[0]);
  out_mem = attention[0]->stages[0].output_mems[0];

  // layer0 mlp module
  d2d(mlp[0]->stages[0].input_mems[0], out_mem);
  net_launch(mlp[0]);
  out_mem = mlp[0]->stages[0].output_mems[0];
  // Save the kv cache: for layer0, get the kv outputs from the mlp_cache module
  d2d(past_key[0], mlp_cache[0]->stages[0].output_mems[1], 0,
      token_length * kv_bytes);
  d2d(past_value[0], mlp_cache[0]->stages[0].output_mems[2], 0,
      token_length * kv_bytes);

  // For layer1 and beyond, execute each layer in order
  for (int idx = 1; idx < NUM_LAYERS; idx++) {
    // Use the attention module
    auto &attn_in = attention[idx]->stages[0].input_mems[0];
    empty(bm_handle, attn_in);
    d2d(attn_in, out_mem, 0, token_length * hidden_bytes);
    // Position and attention mask: only passed in the first time
    if (idx == 1) {
      bm_memcpy_s2d(bm_handle, attention[idx]->stages[0].input_mems[1],
                    (void *)&position_id[0]);
      bm_memcpy_s2d(bm_handle, attention[idx]->stages[0].input_mems[2],
                    (void *)attention_mask.data());
    }
    if (is_dynamic)
      net_launch_dyn(attention[idx]);
    else
      net_launch(attention[idx]);
    out_mem = attention[idx]->stages[0].output_mems[0];

    // Next: the shared moe module
    auto &smoe_in = shared_moe[idx - 1]->stages[0].input_mems[0];
    empty(bm_handle, smoe_in);
    d2d(smoe_in, out_mem, 0, token_length * hidden_bytes);
    if (is_dynamic)
      net_launch_dyn(shared_moe[idx - 1]);
    else
      net_launch(shared_moe[idx - 1]);
    out_mem = shared_moe[idx - 1]->stages[0].output_mems[0];

    // Next: the moe module
    auto &moe_in = moe[idx - 1]->stages[0].input_mems[0];
    empty(bm_handle, moe_in);
    d2d(moe_in, out_mem, 0, token_length * hidden_bytes);
    if (is_dynamic)
      net_launch_dyn(moe[idx - 1]);
    else
      net_launch(moe[idx - 1]);
    out_mem = moe[idx - 1]->stages[0].output_mems[0];

    // Save the current layer's kv cache, obtained from shared_moe_cache (or
    // moe_cache; either one works here)
    d2d(past_key[idx], shared_moe_cache[idx - 1]->stages[0].output_mems[1], 0,
        token_length * kv_bytes);
    d2d(past_value[idx], shared_moe_cache[idx - 1]->stages[0].output_mems[2], 0,
        token_length * kv_bytes);
  }

  // forward lmhead: copy the hidden state of the last token
  auto &lm_in_mem = net_lm->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (token_length - 1) * hidden_bytes, hidden_bytes);
  net_launch(net_lm);

  int token = 0;
  if (generation_mode == "greedy") {
    token = greedy_search(net_greedy_head, lm_out_mem);
  } else if (generation_mode == "penalty_sample") {
    token = penalty_sample(net_penalty_sample_head, lm_out_mem);
  }

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

int Model::forward_next() {
  int cur_token = visited_tokens[token_length - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = token_length - 1;
  // embedding cache
  auto in_mem = net_embed_cache->stages[0].input_mems[0];
  auto out_mem = net_embed_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)&cur_token);
  net_launch(net_embed_cache);

  // For layer0
  {
    auto &in0_mem = attention_cache[0]->stages[0].input_mems[0];
    d2d(in0_mem, out_mem);
    // For the layer0 cache, position and attention mask only need to be copied the first time
    bm_memcpy_s2d(bm_handle, attention_cache[0]->stages[0].input_mems[1],
                  (void *)&position_id);
    bm_memcpy_s2d(bm_handle, attention_cache[0]->stages[0].input_mems[2],
                  (void *)attention_mask.data());
    if (is_dynamic)
      net_launch_dyn(attention_cache[0]);
    else
      net_launch(attention_cache[0]);
    out_mem = attention_cache[0]->stages[0].output_mems[0];
    // layer0 mlp cache
    d2d(mlp_cache[0]->stages[0].input_mems[0], out_mem);
    net_launch(mlp_cache[0]);
    out_mem = mlp_cache[0]->stages[0].output_mems[0];
    int token_offset = (token_length - 1) * kv_bytes;
    bm_memcpy_d2d_byte(bm_handle, past_key[0], token_offset,
                       mlp_cache[0]->stages[0].output_mems[1], 0, kv_bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[0], token_offset,
                       mlp_cache[0]->stages[0].output_mems[2], 0, kv_bytes);
  }

  // For layer1 and beyond
  for (int idx = 1; idx < NUM_LAYERS; idx++) {
    // attention cache
    auto &in0_mem = attention_cache[idx]->stages[0].input_mems[0];
    d2d(in0_mem, out_mem);
    if (idx == 1) {
      bm_memcpy_s2d(bm_handle, attention_cache[idx]->stages[0].input_mems[1],
                    (void *)&position_id);
      bm_memcpy_s2d(bm_handle, attention_cache[idx]->stages[0].input_mems[2],
                    (void *)attention_mask.data());
    } else {
      d2d(attention_cache[idx]->stages[0].input_mems[1],
          attention_cache[0]->stages[0].input_mems[1]);
      d2d(attention_cache[idx]->stages[0].input_mems[2],
          attention_cache[0]->stages[0].input_mems[2]);
    }
    net_launch(attention_cache[idx]);
    out_mem = attention_cache[idx]->stages[0].output_mems[0];

    // shared moe cache
    auto &in_smoe = shared_moe_cache[idx - 1]->stages[0].input_mems[0];
    d2d(in_smoe, out_mem);
    if (is_dynamic)
      net_launch_dyn(shared_moe_cache[idx - 1]);
    else
      net_launch(shared_moe_cache[idx - 1]);
    out_mem = shared_moe_cache[idx - 1]->stages[0].output_mems[0];

    // moe cache
    auto &in_moe = moe_cache[idx - 1]->stages[0].input_mems[0];
    d2d(in_moe, out_mem);
    if (is_dynamic)
      net_launch_dyn(moe_cache[idx - 1]);
    else
      net_launch(moe_cache[idx - 1]);
    out_mem = moe_cache[idx - 1]->stages[0].output_mems[0];

    int token_offset = (token_length - 1) * kv_bytes;
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], token_offset,
                       shared_moe_cache[idx - 1]->stages[0].output_mems[1], 0,
                       kv_bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], token_offset,
                       shared_moe_cache[idx - 1]->stages[0].output_mems[2], 0,
                       kv_bytes);
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

  visited_tokens[token_length] = token;
  token_length += 1;
  return token;
}

std::vector<int> Model::generate(std::vector<int> &history_tokens, int EOS) {
  if (history_tokens.empty()) {
    printf("Sorry: your question is empty!!\n");
    history_tokens.clear();
    return {};
  }

  // Keep the number of input tokens within SEQLEN-10
  int history_length = history_tokens.size();
  if (history_length > SEQLEN - 10) {
    history_tokens.clear();
    printf("Error: your question is too large!\n");
    return {};
  }

  std::vector<int> result_tokens;
  int token = forward_first(history_tokens);
  while (token != EOS && token_length < SEQLEN &&
         token_length <= history_length + max_new_tokens) {
    result_tokens.emplace_back(token);
    token = forward_next();
  }

  return result_tokens;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Model>(m, "Model")
      .def(pybind11::init<>())
      .def("init", &Model::init)
      .def("forward_first", &Model::forward_first)
      .def("forward_next", &Model::forward_next)
      .def("generate", &Model::generate)
      .def("deinit", &Model::deinit)
      .def_readwrite("SEQLEN", &Model::SEQLEN) // Read by pipeline.py
      .def_readwrite("token_length", &Model::token_length)
      .def_readwrite("temperature", &Model::temperature)
      .def_readwrite("top_p", &Model::top_p)
      .def_readwrite("repeat_penalty", &Model::repeat_penalty)
      .def_readwrite("repeat_last_n", &Model::repeat_last_n)
      .def_readwrite("max_new_tokens", &Model::max_new_tokens)
      .def_readwrite("generation_mode", &Model::generation_mode)
      .def_readwrite("prompt_mode", &Model::prompt_mode);
}
