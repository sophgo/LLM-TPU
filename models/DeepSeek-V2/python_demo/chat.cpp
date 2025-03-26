//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
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
#include <getopt.h>
#include <inttypes.h>
#include <iostream>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdio.h>
#include <vector>
#include <dlfcn.h>
#include "utils.h"

static const uint16_t ATTENTION_MASK = 0xC61C;

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
  // 以下几个辅助函数不变
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  void net_launch_dyn(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset, int size);

  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem);

public:
  int hidden_bytes;
  int kv_bytes;
  int token_length;
  int SEQLEN;     // 从bmodel中读取的真实seqlen
  int NUM_LAYERS; // 总层数
  int TOKEN_LEN;
  bool io_alone;
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

  // 模型各模块：  
  // 第一层使用 attention / mlp 模块  
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;

  std::vector<const bm_net_info_t *> attention;       // layer0 使用 attention_0
  std::vector<const bm_net_info_t *> attention_cache; // layer0 的 cache 对应 attention_cache_0
  std::vector<const bm_net_info_t *> mlp;               // layer0 使用 mlp_0
  std::vector<const bm_net_info_t *> mlp_cache;         // layer0 的 cache 对应 mlp_cache_0

  // 第二层及之后，采用 shared moe 结构
  std::vector<const bm_net_info_t *> shared_moe;
  std::vector<const bm_net_info_t *> shared_moe_cache;
  std::vector<const bm_net_info_t *> moe;
  std::vector<const bm_net_info_t *> moe_cache;

  // lm_head 及生成头
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
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, bm_mem_get_device_size(src));
}

void Model::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset, int size) {
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

  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");

  // 获取 embedding 与 lm 模块
  // net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  // net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  // net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  // net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
  // net_penalty_sample_head = bmrt_get_network_info(p_bmrt, "penalty_sample_head");

  // 根据 embedding 层定义真实 SEQLEN
  SEQLEN = 512; // real seqlen

  // 计算 NUM_LAYERS，由bmodel中除去固定模块后的数量决定
  // 注意：总层数的计算公式需保证 layer0 使用 mlp 模块，其余层为 shared moe 结构
  // auto num_nets = bmrt_get_network_number(p_bmrt);
  // 假设 bmodel 中 layer0 有 2模块（attention_0, mlp_0）及其 cache，
  // 其余层每层有 4模块（attention, shared_moe, moe, 相应 cache），另外再减去 embedding、lmhead、head 模块。
  // 这里使用下面公式计算，仅为示例，实际情况可能需要调整：
  NUM_LAYERS = 2;

  // resize visited_tokens
  visited_tokens.resize(SEQLEN);

  // 处理 layer0：attention_0 与 mlp_0
  {
    // attention 模块
    auto attn = bmrt_get_network_info(p_bmrt, "attention_0");
    attention.push_back(attn);
    // attention cache 模块
    auto attn_cache = bmrt_get_network_info(p_bmrt, "attention_cache_0");
    attention_cache.push_back(attn_cache);
    // mlp 模块
    auto mlp_net = bmrt_get_network_info(p_bmrt, "mlp_0");
    mlp.push_back(mlp_net);
    // mlp cache 模块
    auto mlp_cache_net = bmrt_get_network_info(p_bmrt, "mlp_cache_0");
    mlp_cache.push_back(mlp_cache_net);
  }
  // 处理 layer 1 到 NUM_LAYERS-1：shared moe 与 moe 模块
  for (int i = 1; i < NUM_LAYERS; i++) {
    // attention 模块
    auto attn = bmrt_get_network_info(p_bmrt, ("attention_" + std::to_string(i)).c_str());
    attention.push_back(attn);
    // attention cache 模块
    auto attn_cache = bmrt_get_network_info(p_bmrt, ("attention_cache_" + std::to_string(i)).c_str());
    attention_cache.push_back(attn_cache);
    // shared moe 模块
    auto s_moe = bmrt_get_network_info(p_bmrt, ("shared_moe_" + std::to_string(i)).c_str());
    shared_moe.push_back(s_moe);
    // shared moe cache 模块
    auto s_moe_cache = bmrt_get_network_info(p_bmrt, ("shared_moe_cache_" + std::to_string(i)).c_str());
    shared_moe_cache.push_back(s_moe_cache);
    // moe 模块
    auto moe_net = bmrt_get_network_info(p_bmrt, ("moe_" + std::to_string(i)).c_str());
    moe.push_back(moe_net);
    // moe cache 模块
    auto moe_cache_net = bmrt_get_network_info(p_bmrt, ("moe_cache_" + std::to_string(i) + "_0").c_str());
    moe_cache.push_back(moe_cache_net);
  }

  // 设备内存尺寸（按 layer0 mlp_cache 为例）
  hidden_bytes = bm_mem_get_device_size(mlp_cache[0]->stages[0].output_mems[0]);
  kv_bytes = bm_mem_get_device_size(mlp_cache[0]->stages[0].output_mems[1]);

  // kv cache 初始化，与所有层有关（所有层均分配 kv 缓存）
  // 假设所有层都有 kv 缓存，这里统一分配，如果 io_alone 模式则使用 bmodel 中已有内存
  // 总层数为 NUM_LAYERS
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  // 使用 layer0 中任意一个模块判断动态性
  // is_dynamic = attention[0]->is_dynamic;
  // auto addr_mode = attention_cache[0]->addr_mode;
  // io_alone = addr_mode == 1;
  // for (int i = 0; i < NUM_LAYERS; i++) {
  //   assert(addr_mode == attention_cache[i < 1 ? 0 : i-0]->addr_mode);
  //   if (io_alone) {
  //     past_key[i] = (i == 0 ? mlp_cache[0] : shared_moe_cache[i-1])->stages[0].input_mems[3];
  //     past_value[i] = (i == 0 ? mlp_cache[0] : shared_moe_cache[i-1])->stages[0].input_mems[4];
  //   } else {
  //     auto ret = bm_malloc_device_byte(bm_handle, &past_key[i],
  //                                      (i == 0 ? mlp_cache[0] : shared_moe_cache[i-1])->max_input_bytes[3]);
  //     assert(BM_SUCCESS == ret);
  //     ret = bm_malloc_device_byte(bm_handle, &past_value[i],
  //                                 (i == 0 ? mlp_cache[0] : shared_moe_cache[i-1])->max_input_bytes[4]);
  //     assert(BM_SUCCESS == ret);
  //   }
  // }
}

void Model::deinit() {
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
  bm_thread_sync(bm_handle);
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

  int h_bytes = bm_mem_get_device_size(in_tensors[0].device_mem) / SEQLEN;
  bm_set_device_mem(&in_tensors[0].device_mem,
                    h_bytes * TOKEN_LEN,
                    bm_mem_get_device_addr(in_tensors[0].device_mem));
  int pid_bytes = bm_mem_get_device_size(in_tensors[1].device_mem) / SEQLEN;
  bm_set_device_mem(&in_tensors[1].device_mem,
                    pid_bytes * TOKEN_LEN,
                    bm_mem_get_device_addr(in_tensors[1].device_mem));
  int mask_bytes = bm_mem_get_device_size(in_tensors[2].device_mem) / SEQLEN / SEQLEN;
  bm_set_device_mem(&in_tensors[2].device_mem,
                    mask_bytes * TOKEN_LEN * TOKEN_LEN,
                    bm_mem_get_device_addr(in_tensors[2].device_mem));

  in_tensors[0].shape.dims[1] = TOKEN_LEN;
  in_tensors[1].shape.dims[1] = TOKEN_LEN;
  in_tensors[2].shape.dims[2] = TOKEN_LEN;
  in_tensors[2].shape.dims[3] = TOKEN_LEN;

  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

void Model::head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
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

int Model::greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];
  head_launch(net, logits_mem);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

int Model::penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
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
            visited_tokens.begin() + token_length,
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
  // auto duration0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0);
  // std::cout << "net_launch execution time: " << duration0.count() << " ms" << std::endl;


  // auto start = std::chrono::high_resolution_clock::now();
  // for (int i = 0; i < 60; i++) {
  //   net_launch(attention[0]);
  //   net_launch(mlp[0]);
  //   net_launch(shared_moe[0]);
  //   net_launch(moe[0]);
  // }
  // auto end = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  // std::cout << "net_launch execution time: " << duration.count() << " ms" << std::endl;


  // 清空各个模块的输入（此处调用 empty 函数，假设其定义与原来一致）
  // 对于 layer0 的模块：
  empty_net(bm_handle, attention[0]);
  empty_net(bm_handle, mlp[0]);
  empty_net(bm_handle, attention_cache[0]);
  empty_net(bm_handle, mlp_cache[0]);
  // 对于 layer1 及其以后的模块：
  for (int idx = 1; idx < NUM_LAYERS; idx++) {
    empty_net(bm_handle, attention[idx]);
    empty_net(bm_handle, shared_moe[idx-1]);
    empty_net(bm_handle, moe[idx-1]);
    empty_net(bm_handle, attention_cache[idx]);
    empty_net(bm_handle, shared_moe_cache[idx-1]);
    empty_net(bm_handle, moe_cache[idx-1]);
  }

  // forward embedding
  auto &in_mem = net_embed->stages[0].input_mems[0];
  auto &out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)visited_tokens.data());
  net_launch(net_embed); // run embedding

  // 第一层：layer0
  // forward attention_0：这里假设使用 embedding 层输出直接拷贝给 attention 模块
  auto &in0_mem = attention[0]->stages[0].input_mems[0];
  empty(bm_handle, attention[0]->stages[0].input_mems[0]);
  d2d(in0_mem, out_mem, 0, token_length * hidden_bytes);
  // 对于 layer0 attention，只有第一次传入 position 与 attention_mask
  bm_memcpy_s2d(bm_handle, attention[0]->stages[0].input_mems[1], (void *)position_id.data());
  bm_memcpy_s2d(bm_handle, attention[0]->stages[0].input_mems[2], (void *)attention_mask.data());
  if (is_dynamic) net_launch_dyn(attention[0]);
  else net_launch(attention[0]);
  out_mem = attention[0]->stages[0].output_mems[0];

  // layer0 mlp 模块
  d2d(mlp[0]->stages[0].input_mems[0], out_mem);
  net_launch(mlp[0]);
  out_mem = mlp[0]->stages[0].output_mems[0];
  // kv cache 保存：对于 layer0，从 mlp_cache 模块获取 kv 输出
  d2d(past_key[0], mlp_cache[0]->stages[0].output_mems[1], 0, token_length * kv_bytes);
  d2d(past_value[0], mlp_cache[0]->stages[0].output_mems[2], 0, token_length * kv_bytes);

  // 对于 layer1 及以后，每一层顺序执行
  for (int idx = 1; idx < NUM_LAYERS; idx++) {
    // 使用 attention 模块
    auto &attn_in = attention[idx]->stages[0].input_mems[0];
    empty(bm_handle, attn_in);
    d2d(attn_in, out_mem, 0, token_length * hidden_bytes);
    // 位置、attention mask：仅第一次传入
    if (idx == 1) {
      bm_memcpy_s2d(bm_handle, attention[idx]->stages[0].input_mems[1], (void *)&position_id[0]);
      bm_memcpy_s2d(bm_handle, attention[idx]->stages[0].input_mems[2], (void *)attention_mask.data());
    }
    if (is_dynamic) net_launch_dyn(attention[idx]);
    else net_launch(attention[idx]);
    out_mem = attention[idx]->stages[0].output_mems[0];

    // 接下来 shared moe 模块
    auto &smoe_in = shared_moe[idx-1]->stages[0].input_mems[0];
    empty(bm_handle, smoe_in);
    d2d(smoe_in, out_mem, 0, token_length * hidden_bytes);
    if (is_dynamic) net_launch_dyn(shared_moe[idx-1]);
    else net_launch(shared_moe[idx-1]);
    out_mem = shared_moe[idx-1]->stages[0].output_mems[0];

    // 接下来 moe 模块
    auto &moe_in = moe[idx-1]->stages[0].input_mems[0];
    empty(bm_handle, moe_in);
    d2d(moe_in, out_mem, 0, token_length * hidden_bytes);
    if (is_dynamic) net_launch_dyn(moe[idx-1]);
    else net_launch(moe[idx-1]);
    out_mem = moe[idx-1]->stages[0].output_mems[0];

    // 保存当前层的 kv cache，从 shared_moe_cache（或 moe_cache，此处任选其一）获取
    d2d(past_key[idx], shared_moe_cache[idx-1]->stages[0].output_mems[1], 0,
        token_length * kv_bytes);
    d2d(past_value[idx], shared_moe_cache[idx-1]->stages[0].output_mems[2], 0,
        token_length * kv_bytes);
  }

  // forward lmhead：拷贝最后一个 token 的 hidden state
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
  auto &in_mem = net_embed_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)&cur_token);
  net_launch(net_embed_cache);

  // 对于 layer0
  {
    auto &in0_mem = attention_cache[0]->stages[0].input_mems[0];
    d2d(in0_mem, out_mem);
    // 对于 layer0 cache，输入 position 和 attention mask 仅第一次需要拷贝
    bm_memcpy_s2d(bm_handle, attention_cache[0]->stages[0].input_mems[1], (void *)&position_id);
    bm_memcpy_s2d(bm_handle, attention_cache[0]->stages[0].input_mems[2], (void *)attention_mask.data());
    if (is_dynamic) net_launch_dyn(attention_cache[0]);
    else net_launch(attention_cache[0]);
    out_mem = attention_cache[0]->stages[0].output_mems[0];
    // layer0 mlp cache
    d2d(mlp_cache[0]->stages[0].input_mems[0], out_mem);
    net_launch(mlp_cache[0]);
    out_mem = mlp_cache[0]->stages[0].output_mems[0];
    int token_offset = (token_length - 1) * kv_bytes;
    bm_memcpy_d2d_byte(bm_handle, past_key[0], token_offset, mlp_cache[0]->stages[0].output_mems[1],
                       0, kv_bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[0], token_offset, mlp_cache[0]->stages[0].output_mems[2],
                       0, kv_bytes);
  }

  // 对于 layer1 及之后
  for (int idx = 1; idx < NUM_LAYERS; idx++) {
    // attention cache
    auto &in0_mem = attention_cache[idx]->stages[0].input_mems[0];
    d2d(in0_mem, out_mem);
    if (io_alone) {
      if (idx == 1) {
        bm_memcpy_s2d(bm_handle, attention_cache[idx]->stages[0].input_mems[1], (void *)&position_id);
        bm_memcpy_s2d(bm_handle, attention_cache[idx]->stages[0].input_mems[2], (void *)attention_mask.data());
      } else {
        d2d(attention_cache[idx]->stages[0].input_mems[1],
            attention_cache[0]->stages[0].input_mems[1]);
        d2d(attention_cache[idx]->stages[0].input_mems[2],
            attention_cache[0]->stages[0].input_mems[2]);
      }
    } else {
      if (idx == 1) {
        bm_memcpy_s2d(bm_handle, attention_cache[idx]->stages[0].input_mems[1], (void *)&position_id);
        bm_memcpy_s2d(bm_handle, attention_cache[idx]->stages[0].input_mems[2], (void *)attention_mask.data());
      }
      d2d(attention_cache[idx]->stages[0].input_mems[3], past_key[idx]);
      d2d(attention_cache[idx]->stages[0].input_mems[4], past_value[idx]);
    }
    net_launch(attention_cache[idx]);
    out_mem = attention_cache[idx]->stages[0].output_mems[0];

    // shared moe cache
    auto &in_smoe = shared_moe_cache[idx-1]->stages[0].input_mems[0];
    d2d(in_smoe, out_mem);
    if (is_dynamic) net_launch_dyn(shared_moe_cache[idx-1]);
    else net_launch(shared_moe_cache[idx-1]);
    out_mem = shared_moe_cache[idx-1]->stages[0].output_mems[0];

    // moe cache
    auto &in_moe = moe_cache[idx-1]->stages[0].input_mems[0];
    d2d(in_moe, out_mem);
    if (is_dynamic) net_launch_dyn(moe_cache[idx-1]);
    else net_launch(moe_cache[idx-1]);
    out_mem = moe_cache[idx-1]->stages[0].output_mems[0];

    int token_offset = (token_length - 1) * kv_bytes;
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], token_offset,
                       shared_moe_cache[idx-1]->stages[0].output_mems[1],
                       0, kv_bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], token_offset,
                       shared_moe_cache[idx-1]->stages[0].output_mems[2],
                       0, kv_bytes);
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

  // 控制输入 token 数不超过 SEQLEN-10
  int history_length = history_tokens.size();
  if (history_length > SEQLEN - 10) {
    history_tokens.clear();
    printf("Error: your question is too large!\n");
    return {};
  }

  std::vector<int> result_tokens;
  int token = forward_first(history_tokens);
  while (token != EOS && token_length < SEQLEN && token_length <= history_length + max_new_tokens) {
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
      .def_readwrite("SEQLEN", &Model::SEQLEN) // 供 pipeline.py 读取
      .def_readwrite("token_length", &Model::token_length)
      .def_readwrite("temperature", &Model::temperature)
      .def_readwrite("top_p", &Model::top_p)
      .def_readwrite("repeat_penalty", &Model::repeat_penalty)
      .def_readwrite("repeat_last_n", &Model::repeat_last_n)
      .def_readwrite("max_new_tokens", &Model::max_new_tokens)
      .def_readwrite("generation_mode", &Model::generation_mode)
      .def_readwrite("prompt_mode", &Model::prompt_mode);
}
