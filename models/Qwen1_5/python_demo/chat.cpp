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

static const uint16_t ATTENTION_MASK = 0xF0E2;

typedef union {
  float fval;
  uint32_t bits;
  struct {
    uint32_t frac : 23; // mantissa
    uint32_t exp : 8;   // exponent
    uint32_t sign : 1;  // sign
  } format;
} fp32;

static inline uint16_t fp32_value_to_fp16_ieee(float fval) {
  // 将fp32的float类型的值转换为位表示
  fp32 u;
  u.fval = fval;

  // 提取fp32的符号、指数和尾数
  uint32_t sign = u.format.sign;
  int32_t exp = u.format.exp - 127; // fp32的偏移指数
  uint32_t frac = u.format.frac;

  // 初始化fp16的符号、指数和尾数
  uint16_t h_sign = sign << 15;
  uint16_t h_exp, h_frac;

  // 检查是否为0
  if (exp == -127) {
    // 输入为0或接近0的数，直接返回0
    return h_sign;
  }

  // 检查是否为NaN或无穷大
  if (exp > 15) {
    // 指数太大，无法表示为fp16，返回无穷大
    h_exp = 0x1F;
    h_frac = 0; // 如果是NaN，可以设置不为0的尾数
  } else if (exp > -15) {
    // 正常范围，可以转换为fp16
    exp += 15; // 调整指数的偏移量
    h_exp = exp << 10;
    h_frac = frac >> 13; // 缩减尾数
  } else {
    // 指数太小，数值太小，无法表示为规格化的fp16
    h_exp = 0;
    h_frac = 0; // 可能需要处理subnormals
  }

  // 组合符号位、指数和尾数
  return h_sign | h_exp | h_frac;
}

class Qwen {
public:
  void init(const std::vector<int> &devid, std::string model_path,
            const float &__temperature, const float &__top_p,
            const int &__max_new_tokens, const std::string &__generation_mode,
            const std::string &__prompt_mode);
  void deinit();
  int forward_first(std::vector<int> &tokens);
  int forward_next();

  std::mt19937 sgen;
  Qwen() : sgen(std::random_device()()){};
  int sample(const std::vector<float> &probs, const std::vector<int> &tokens);

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
  int token_length;
  int SEQLEN;     // read from bmodel
  int NUM_LAYERS; // read from bmodel
  bool io_alone;

  // generation
  float temperature;
  uint16_t top_p;
  float repeat_penalty;
  int repeat_last_n;
  int max_new_tokens;
  std::string generation_mode;
  std::string prompt_mode;
  std::vector<int> visited_tokens;
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
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void Qwen::init(const std::vector<int> &devices, std::string model_path,
                const float &__temperature, const float &__top_p,
                const int &__max_new_tokens,
                const std::string &__generation_mode,
                const std::string &__prompt_mode) {
  // generation params
  temperature = __temperature;
  top_p = fp32_value_to_fp16_ieee(__top_p);
  max_new_tokens = __max_new_tokens;
  generation_mode = __generation_mode;
  prompt_mode = __prompt_mode;

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
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1]; // real seqlen
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = (num_nets - 2) / 2;

  // resize
  visited_tokens.resize(SEQLEN);

  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  auto addr_mode = net_blocks_cache[0]->addr_mode;
  io_alone = addr_mode == 1;
  for (int i = 0; i < NUM_LAYERS; i++) {
    assert(addr_mode == net_blocks_cache[i]->addr_mode);
    if (io_alone) {
      past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
      past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    } else {
      auto ret = bm_malloc_device_byte(bm_handle, &past_key[i],
                                       net_blocks_cache[i]->max_input_bytes[3]);
      assert(BM_SUCCESS == ret);
      ret = bm_malloc_device_byte(bm_handle, &past_value[i],
                                  net_blocks_cache[i]->max_input_bytes[4]);
      assert(BM_SUCCESS == ret);
    }
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

int Qwen::sample(const std::vector<float> &probs,
                 const std::vector<int> &tokens) {
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return tokens[dist(sgen)];
}

int Qwen::forward_first(std::vector<int> &tokens) {
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());

  token_length = tokens.size();

  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < SEQLEN; j++) {
      if (j <= i) {
        attention_mask[i * SEQLEN + j] = 0;
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
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2]);
  }

  int bytes = out_mem.size / SEQLEN;

  auto &lm_in0_mem = net_lm->stages[0].input_mems[0];
  auto &lm_in1_mem = net_lm->stages[0].input_mems[1];
  auto &lm_in2_mem = net_lm->stages[0].input_mems[2];
  auto &lm_out_logits_mem = net_lm->stages[0].output_mems[0];
  auto &lm_out_tokens_mem = net_lm->stages[0].output_mems[1];

  // top_p + top_k + temperature
  bm_memcpy_d2d_byte(bm_handle, lm_in0_mem, 0, out_mem,
                     (token_length - 1) * bytes, bytes);
  bm_memcpy_s2d(bm_handle, lm_in1_mem, (void *)&top_p);
  bm_memcpy_s2d(bm_handle, lm_in2_mem, (void *)&temperature);
  net_launch(net_lm);

  // get logit & token
  int candidate_num = net_lm->stages[0].output_shapes[0].dims[1];
  std::vector<float> lm_logits(candidate_num);
  bm_memcpy_d2s(bm_handle, lm_logits.data(), lm_out_logits_mem);
  std::vector<int> lm_tokens(candidate_num);
  bm_memcpy_d2s(bm_handle, lm_tokens.data(), lm_out_tokens_mem);

  // process the lookahead tokens
  int token;
  if (generation_mode == "greedy") {
    token = lm_tokens[0];
  } else if (generation_mode == "sample") {
    token = sample(lm_logits, lm_tokens);
  }

  visited_tokens.emplace_back(token);
  return token;
}

int Qwen::forward_next() {
  token_length += 1;
  int cur_token = visited_tokens[visited_tokens.size() - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = token_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = token_length - 1;
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
    auto &in3_mem = net_blocks_cache[idx]->stages[0].input_mems[3];
    auto &in4_mem = net_blocks_cache[idx]->stages[0].input_mems[4];
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
      if (idx == 0) {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      }
      d2d(in3_mem, past_key[idx]);
      d2d(in4_mem, past_value[idx]);
    }
    net_launch(net_blocks_cache[idx]);
    out_mem = out0_mem;
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], token_offset, out1_mem, 0,
                       bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], token_offset, out2_mem, 0,
                       bytes);
  }

  auto &lm_in0_mem = net_lm->stages[0].input_mems[0];
  auto &lm_in1_mem = net_lm->stages[0].input_mems[1];
  auto &lm_in2_mem = net_lm->stages[0].input_mems[2];
  auto &lm_out_logits_mem = net_lm->stages[0].output_mems[0];
  auto &lm_out_tokens_mem = net_lm->stages[0].output_mems[1];

  // repeat_penalty + top_p + top_k + temperature
  d2d(lm_in0_mem, out_mem);
  bm_memcpy_s2d(bm_handle, lm_in1_mem, (void *)&top_p);
  bm_memcpy_s2d(bm_handle, lm_in2_mem, (void *)&temperature);
  net_launch(net_lm);

  int candidate_num = net_lm->stages[0].output_shapes[0].dims[1];
  std::vector<float> lm_logits(candidate_num);
  bm_memcpy_d2s(bm_handle, lm_logits.data(), lm_out_logits_mem);
  std::vector<int> lm_tokens(candidate_num);
  bm_memcpy_d2s(bm_handle, lm_tokens.data(), lm_out_tokens_mem);

  // process the lookahead tokens
  int token;
  if (generation_mode == "greedy") {
    token = lm_tokens[0];
  } else if (generation_mode == "sample") {
    token = sample(lm_logits, lm_tokens);
  }

  visited_tokens.emplace_back(token);
  return token;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Qwen>(m, "Qwen")
      .def(pybind11::init<>())
      .def("init", &Qwen::init)
      .def("forward_first", &Qwen::forward_first)
      .def("forward_next", &Qwen::forward_next)
      .def("deinit", &Qwen::deinit);
}
