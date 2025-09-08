//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "memory.h"
#include "tpuv7_modelrt.h"
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

namespace py = pybind11;
//===------------------------------------------------------------===//
// Helper Func
//===------------------------------------------------------------===//
/* get data type byte size */
static inline int DtypeSize(const tpuRtDataType_t &dtype) {
  switch (dtype) {
  case TPU_FLOAT32:
  case TPU_INT32:
  case TPU_UINT32:
    return 4;
  case TPU_FLOAT16:
  case TPU_BFLOAT16:
  case TPU_INT16:
    return 2;
  case TPU_INT8:
  case TPU_UINT8:
    return 1;
  case TPU_INT4:
  case TPU_UINT4:
    return 1; // need modify ?  to do
  default:
    return 1;
  }
  return 0;
}

/* number of shape elements */
static inline uint64_t ShapeCount(const tpuRtShape_t &shape) {
  uint64_t count = 1;
  for (int i = 0; i < shape.num_dims; i++) {
    count *= shape.dims[i];
  }
  return count;
}

static void empty(void *devPtr, uint64_t size, tpuRtStream_t m_stream) {
  // int value = 0;
  // auto ret = tpuRtMemsetAsync(devPtr, value, size, m_stream);
  uint8_t *zero = new uint8_t[size];
  memset(zero, 0, size);
  auto ret = tpuRtMemcpyS2DAsync(devPtr, zero, size, m_stream);
  delete[] zero;
  assert(tpuRtSuccess == ret);
}

static void empty_net(const tpuRtNetInfo_t &net, tpuRtStream_t m_stream) {
  for (int i = 0; i < net.input.num; i++) {
    uint64_t size = ShapeCount(net.stages[0].input_shapes[i]) *
                    DtypeSize(net.input.dtypes[i]);
    empty(net.stages[0].input_mems[i], size, m_stream);
  }
  for (int i = 0; i < net.output.num; i++) {
    uint64_t size = ShapeCount(net.stages[0].output_shapes[i]) *
                    DtypeSize(net.output.dtypes[i]);
    empty(net.stages[0].output_mems[i], size, m_stream);
  }
}

static void getIOTensor(std::vector<tpuRtTensor_t> &in_tensors,
                        std::vector<tpuRtTensor_t> &out_tensors,
                        const tpuRtNetInfo_t &net, int stage_idx = 0) {
  auto mem = net.stages[stage_idx].input_mems;
  auto shape = net.stages[stage_idx].input_shapes;
  for (int i = 0; i < net.input.num; ++i) {
    in_tensors[i].dtype = net.input.dtypes[i];
    in_tensors[i].shape.num_dims = shape[i].num_dims;
    memcpy(in_tensors[i].shape.dims, shape[i].dims,
           sizeof(int) * shape[i].num_dims);
    in_tensors[i].data = mem[i];
  }
  mem = net.stages[stage_idx].output_mems;
  shape = net.stages[stage_idx].output_shapes;
  for (int i = 0; i < net.output.num; ++i) {
    out_tensors[i].dtype = net.output.dtypes[i];
    out_tensors[i].shape.num_dims = shape[i].num_dims;
    memcpy(out_tensors[i].shape.dims, shape[i].dims,
           sizeof(int) * shape[i].num_dims);
    out_tensors[i].data = mem[i];
  }
}

class Qwen {
public:
  void init(int devid, std::string model_path);
  void deinit();
  int forward_first(std::vector<int> &tokens);
  int forward_next();
  void clear_kv();
  std::vector<int> generate(std::vector<int> &history_tokens, int EOS);

  std::mt19937 sgen;
  Qwen() : sgen(std::random_device()()) {};

private:
  int forward_first_with_kv(std::vector<int> &tokens);
  void net_launch(const tpuRtNetInfo_t &net);
  void net_launch_dyn(const tpuRtNetInfo_t &net, int real_len);
  void net_launch_decode(int block_idx, int kv_offset, void *input_mem,
                         const int *position_id,
                         std::vector<uint16_t> &attention_mask);
  void init_by_names();

public:
  int hidden_bytes;
  int kv_bytes;
  int token_length;
  int SEQLEN;
  int MAX_INPUT_LENGTH;
  int PREFILL_KV_LENGTH;
  int NUM_LAYERS;
  bool lmhead_with_topk;
  bool is_dynamic;
  std::vector<int> visited_tokens;
  bool support_prefill_kv;
  int history_length;
  uint16_t mask_value;

  // generation
  std::string generation_mode;
  float penalty;
  float temperature;
  int top_k;
  float top_p;

private:
  tpuRtNetContext_t m_context;
  tpuRtStream_t m_stream;
  tpuRtNet_t m_net;
  std::vector<tpuRtNetInfo_t> net_blocks;
  std::vector<tpuRtNetInfo_t> net_blocks_cache;
  tpuRtNetInfo_t net_embed;
  tpuRtNetInfo_t net_embed_cache;
  tpuRtNetInfo_t net_lm, net_greedy_head, net_sample_head;
  void *dev_buffer;
  uint64_t dev_buffer_size;
  uint64_t kv_buffer_size;
  std::vector<void *> past_key;
  std::vector<void *> past_value;
};

void Qwen::init_by_names() {
  auto is_exist = [](const char *name, char **names, int num) {
    for (int i = 0; i < num; i++) {
      if (strcmp(name, names[i]) == 0) {
        return true;
      }
    }
    return false;
  };
  net_embed = tpuRtGetNetInfo(m_net, "embedding");
  net_embed_cache = tpuRtGetNetInfo(m_net, "embedding_cache");
  net_lm = tpuRtGetNetInfo(m_net, "lm_head");
  char **net_names = nullptr;
  auto num_nets = tpuRtGetNetNames(m_net, &net_names);
  auto num_blocks = num_nets - 3; // 3 nets are embed, lm_head, embedding_cache
  if (is_exist("greedy_head", net_names, num_nets)) {
    net_greedy_head = tpuRtGetNetInfo(m_net, "greedy_head");
    num_blocks--; // greedy_head is not a block
  }
  if (is_exist("sample_head", net_names, num_nets)) {
    net_sample_head = tpuRtGetNetInfo(m_net, "sample_head");
    num_blocks--; // sample_head is not a block
  }

  lmhead_with_topk = net_lm.stages[0].output_shapes[0].dims[1] == 1;

  NUM_LAYERS = num_blocks / 2; // 2 nets for each block, one for cache
  // net blocks
  for (int i = 0; i < num_blocks / 2; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    if ((!is_exist(block_name.c_str(), net_names, num_nets)) ||
        (!is_exist(cache_name.c_str(), net_names, num_nets))) {
      NUM_LAYERS = i;
      printf("Warning: Only %d blocks found, expected %d blocks.\n", NUM_LAYERS,
             num_blocks / 2);
      break;
    }
    net_blocks.emplace_back(tpuRtGetNetInfo(m_net, block_name.c_str()));
    net_blocks_cache.emplace_back(tpuRtGetNetInfo(m_net, cache_name.c_str()));
  }
  tpuRtFreeNetNames(net_names);
  if (net_embed_cache.output.dtypes[0] == TPU_FLOAT16) {
    mask_value = 0xF0E2; // float16
  } else if (net_embed_cache.output.dtypes[0] == TPU_BFLOAT16) {
    mask_value = 0xC61C; // -9984 by bfloat16
  } else {
    std::cerr << "\nError: Invalid attention dtype\n";
    std::cerr << "Supported dtype are 'BM_FLOAT16' or 'BM_BFLOAT16'\n";
    throw std::runtime_error("Invalid attention dtype");
  }
  MAX_INPUT_LENGTH = net_embed.stages[0].input_shapes[0].dims[1];
  SEQLEN = net_blocks_cache[0].stages[0].input_shapes[3].dims[1];
  support_prefill_kv = net_blocks[0].input.num == 5; // with kv cache
  history_length = 0;
  printf("Num Layers:%d\n", NUM_LAYERS);
  if (support_prefill_kv) {
    PREFILL_KV_LENGTH = net_blocks[0].stages[0].input_shapes[3].dims[1];
    printf("History by kv: True\n");
  }
}

void Qwen::init(int device, std::string model_path) {

  // request bm_handle
  std::cout << "Device [ " << device << " ] \n";
  auto ret = tpuRtInit();
  assert(tpuRtSuccess == ret);
  ret = tpuRtSetDevice(device);
  assert(tpuRtSuccess == ret);

  ret = tpuRtCreateNetContext(&m_context);
  assert(tpuRtSuccess == ret);
  ret = tpuRtStreamCreate(&m_stream);
  assert(tpuRtSuccess == ret);

  std::cout << "Model [" << model_path.c_str() << "] loading .... ";
  ret = tpuRtLoadNet(model_path.c_str(), m_context, &m_net);
  assert(tpuRtSuccess == ret);
  std::cout << "Done!" << std::endl;

  init_by_names();
  visited_tokens.resize(SEQLEN);
  hidden_bytes = ShapeCount(net_blocks_cache[0].stages[0].output_shapes[0]) *
                 DtypeSize(net_blocks_cache[0].output.dtypes[0]);
  kv_bytes = ShapeCount(net_blocks_cache[0].stages[0].output_shapes[1]) *
             DtypeSize(net_blocks_cache[0].output.dtypes[1]);
  dev_buffer_size = ShapeCount(net_embed.stages[0].output_shapes[0]) *
                    DtypeSize(net_embed.output.dtypes[0]);

  ret = tpuRtMalloc(&dev_buffer, dev_buffer_size, 1);
  assert(tpuRtSuccess == ret);
  empty(dev_buffer, dev_buffer_size, m_stream);

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  is_dynamic = net_blocks[0].is_dynamic;
  kv_buffer_size = ShapeCount(net_blocks_cache[0].stages[0].input_shapes[3]) *
                   DtypeSize(net_blocks_cache[0].input.dtypes[3]);
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i] = net_blocks_cache[i].stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i].stages[0].input_mems[4];
    empty(past_key[i], kv_buffer_size, m_stream);
    empty(past_value[i], kv_buffer_size, m_stream);
  }
}

void Qwen::deinit() {
  tpuRtFree(&dev_buffer, 1);
  tpuRtDestroyNetContext(m_context);
  tpuRtStreamDestroy(m_stream);
}

void Qwen::net_launch(const tpuRtNetInfo_t &net) {
  std::vector<tpuRtTensor_t> in_tensors(net.input.num);
  std::vector<tpuRtTensor_t> out_tensors(net.output.num);
  getIOTensor(in_tensors, out_tensors, net);
  auto ret = tpuRtLaunchNetAsync(m_net, in_tensors.data(), out_tensors.data(),
                                 net.name, m_stream);
  assert(tpuRtSuccess == ret);
}

void Qwen::net_launch_dyn(const tpuRtNetInfo_t &net, int real_len) {
  std::vector<tpuRtTensor_t> in_tensors(net.input.num);
  std::vector<tpuRtTensor_t> out_tensors(net.output.num);
  getIOTensor(in_tensors, out_tensors, net);

  in_tensors[0].shape.dims[1] = real_len;
  in_tensors[1].shape.dims[1] = real_len;
  in_tensors[2].shape.dims[2] = real_len;
  in_tensors[2].shape.dims[3] = real_len;

  auto ret = tpuRtLaunchNetAsync(m_net, in_tensors.data(), out_tensors.data(),
                                 net.name, m_stream);
  assert(tpuRtSuccess == ret);
}

void Qwen::net_launch_decode(int idx, int kv_offset, void *input_mem,
                             const int *position_id,
                             std::vector<uint16_t> &attention_mask) {
  auto &net = net_blocks_cache[idx];
  assert(net.input.num == 5 && net.output.num == 3);
  std::vector<tpuRtTensor_t> in_tensors(net.input.num);
  std::vector<tpuRtTensor_t> out_tensors(net.output.num);
  getIOTensor(in_tensors, out_tensors, net);

  // ===== adjust input tensors =====
  in_tensors[0].data = input_mem;
  if (idx == 0) {
    tpuRtMemcpyS2DAsync(in_tensors[1].data, (void *)position_id, sizeof(int),
                        m_stream);
    tpuRtMemcpyS2DAsync(in_tensors[2].data, (void *)attention_mask.data(),
                        attention_mask.size() * sizeof(uint16_t), m_stream);
  } else {
    in_tensors[1].data = net_blocks_cache[0].stages[0].input_mems[1];
    in_tensors[2].data = net_blocks_cache[0].stages[0].input_mems[2];
  }
  // ===== adjust output tensors =====
  out_tensors[1].data = (void *)((uint8_t *)in_tensors[3].data + kv_offset);
  out_tensors[2].data = (void *)((uint8_t *)in_tensors[4].data + kv_offset);
  // ===== launch =====
  auto ret = tpuRtLaunchNetAsync(m_net, in_tensors.data(), out_tensors.data(),
                                 net.name, m_stream);
  assert(tpuRtSuccess == ret);
}

int Qwen::forward_first(std::vector<int> &tokens) {
  if (support_prefill_kv) {
    return forward_first_with_kv(tokens);
  }
  std::vector<int> position_id(MAX_INPUT_LENGTH, 0);
  std::vector<uint16_t> attention_mask(MAX_INPUT_LENGTH * MAX_INPUT_LENGTH,
                                       mask_value);
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());

  token_length = tokens.size();

  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }
  if (is_dynamic) {
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * token_length + j] = 0;
      }
    }
  } else {
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * MAX_INPUT_LENGTH + j] = 0;
      }
    }
  }

  // forward embeding
  auto in_mem = net_embed.stages[0].input_mems[0];
  auto out_mem = net_embed.stages[0].output_mems[0];
  tpuRtMemcpyS2DAsync(in_mem, (void *)visited_tokens.data(),
                      MAX_INPUT_LENGTH * sizeof(int), m_stream);

  net_launch(net_embed);
  tpuRtMemcpyD2DAsync(dev_buffer, out_mem, MAX_INPUT_LENGTH * hidden_bytes,
                      m_stream);
  out_mem = dev_buffer;

  // forward blocks
  empty_net(net_blocks[0], m_stream);
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx].stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx].stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx].stages[0].input_mems[2];
    tpuRtMemcpyD2DAsync(in0_mem, out_mem, token_length * hidden_bytes,
                        m_stream);
    if (idx == 0) {
      // only first time need copy
      tpuRtMemcpyS2DAsync(in1_mem, (void *)position_id.data(),
                          token_length * sizeof(int), m_stream);
      tpuRtMemcpyS2DAsync(
          in2_mem, (void *)attention_mask.data(),
          MAX_INPUT_LENGTH * MAX_INPUT_LENGTH * sizeof(uint16_t), m_stream);
    }
    if (is_dynamic) {
      net_launch_dyn(net_blocks[idx], token_length);
    } else {
      net_launch(net_blocks[idx]);
    }
    out_mem = net_blocks[idx].stages[0].output_mems[0];
    tpuRtMemcpyD2DAsync(past_key[idx], net_blocks[idx].stages[0].output_mems[1],
                        token_length * kv_bytes, m_stream);
    tpuRtMemcpyD2DAsync(past_value[idx],
                        net_blocks[idx].stages[0].output_mems[2],
                        token_length * kv_bytes, m_stream);
  }

  // forward lmhead
  auto &lm_in_mem = net_lm.stages[0].input_mems[0];
  auto &lm_out_mem = net_lm.stages[0].output_mems[0];
  tpuRtMemcpyD2DAsync(
      lm_in_mem,
      (void *)((uint8_t *)out_mem + (token_length - 1) * hidden_bytes),
      hidden_bytes, m_stream);
  net_launch(net_lm);

  int token = 0;
  tpuRtMemcpyD2SAsync((void *)&token, lm_out_mem, sizeof(int), m_stream);
  tpuRtStreamSynchronize(m_stream);

  visited_tokens[token_length] = token;
  token_length += 1;
  history_length = token_length;
  return token;
}

int Qwen::forward_first_with_kv(std::vector<int> &inputs) {
  int max_kv_length = MAX_INPUT_LENGTH + PREFILL_KV_LENGTH;
  std::vector<int> position_id(MAX_INPUT_LENGTH, 0);
  std::copy(inputs.begin(), inputs.end(), visited_tokens.data());
  auto old_length = history_length;
  token_length = inputs.size();
  history_length += token_length;
  std::vector<uint16_t> attention_mask(MAX_INPUT_LENGTH * max_kv_length,
                                       mask_value);
  assert(history_length < SEQLEN);
  assert(old_length <= PREFILL_KV_LENGTH);
  for (int i = 0; i < token_length; i++) {
    for (int j = 0; j < old_length; j++) {
      attention_mask[i * max_kv_length + j] = 0;
    }
    for (int j = 0; j <= i; j++) {
      attention_mask[i * max_kv_length + j + PREFILL_KV_LENGTH] = 0;
    }
  }
  for (int i = 0; i < token_length; i++) {
    position_id[i] = i + old_length;
  }
  // forward embeding
  auto in_mem = net_embed.stages[0].input_mems[0];
  auto out_mem = net_embed.stages[0].output_mems[0];
  empty_net(net_embed, m_stream);
  tpuRtMemcpyS2DAsync(in_mem, (void *)inputs.data(), token_length * sizeof(int),
                      m_stream);
  net_launch(net_embed);
  tpuRtMemcpyD2DAsync(dev_buffer, out_mem, MAX_INPUT_LENGTH * hidden_bytes,
                      m_stream);
  out_mem = dev_buffer;

  // forward blocks
  empty_net(net_blocks[0], m_stream);
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx].stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx].stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx].stages[0].input_mems[2];
    auto &in3_mem = net_blocks[idx].stages[0].input_mems[3];
    auto &in4_mem = net_blocks[idx].stages[0].input_mems[4];
    tpuRtMemcpyD2DAsync(in0_mem, out_mem, token_length * hidden_bytes,
                        m_stream);
    if (old_length > 0) {
      tpuRtMemcpyD2DAsync(in3_mem, past_key[idx], kv_bytes * old_length,
                          m_stream);
      tpuRtMemcpyD2DAsync(in4_mem, past_value[idx], kv_bytes * old_length,
                          m_stream);
    } else if (idx == 0) {
      empty(in3_mem, PREFILL_KV_LENGTH * kv_bytes, m_stream);
      empty(in4_mem, PREFILL_KV_LENGTH * kv_bytes, m_stream);
    }
    tpuRtMemcpyS2DAsync(in1_mem, (void *)position_id.data(),
                        MAX_INPUT_LENGTH * sizeof(int), m_stream);
    tpuRtMemcpyS2DAsync(in2_mem, (void *)attention_mask.data(),
                        attention_mask.size() * sizeof(uint16_t), m_stream);
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx].stages[0].output_mems[0];
    auto &out1_mem = net_blocks[idx].stages[0].output_mems[1];
    auto &out2_mem = net_blocks[idx].stages[0].output_mems[2];
    tpuRtMemcpyD2DAsync(
        (void *)((uint8_t *)past_key[idx] + old_length * kv_bytes), out1_mem,
        token_length * kv_bytes, m_stream);
    tpuRtMemcpyD2DAsync(
        (void *)((uint8_t *)past_value[idx] + old_length * kv_bytes), out2_mem,
        token_length * kv_bytes, m_stream);
  }

  // forward lmhead
  auto &lm_in_mem = net_lm.stages[0].input_mems[0];
  auto &lm_out_mem = net_lm.stages[0].output_mems[0];
  tpuRtMemcpyD2DAsync(
      lm_in_mem,
      (void *)((uint8_t *)out_mem + (token_length - 1) * hidden_bytes),
      hidden_bytes, m_stream);
  net_launch(net_lm);
  int token = 0;
  tpuRtMemcpyD2SAsync((void *)&token, lm_out_mem, sizeof(int), m_stream);
  tpuRtStreamSynchronize(m_stream);
  visited_tokens[token_length] = token;
  token_length++;
  history_length++;
  return token;
}

int Qwen::forward_next() {
  int cur_token = visited_tokens[token_length - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = history_length - 1; i < SEQLEN; i++) {
    attention_mask[i] = mask_value;
  }
  int32_t position_id = history_length - 1;
  // embedding
  auto in_mem = net_embed_cache.stages[0].input_mems[0];
  auto out_mem = net_embed_cache.stages[0].output_mems[0];
  tpuRtMemcpyS2DAsync(in_mem, (void *)&cur_token, sizeof(int), m_stream);
  net_launch(net_embed_cache);

  // blocks
  int token_offset = (token_length - 1) * kv_bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    net_launch_decode(idx, token_offset, out_mem, &position_id, attention_mask);
    out_mem = net_blocks_cache[idx].stages[0].output_mems[0];
  }

  // forward lmhead
  auto &lm_in_mem = net_lm.stages[0].input_mems[0];
  auto &lm_out_mem = net_lm.stages[0].output_mems[0];
  tpuRtMemcpyD2DAsync(lm_in_mem, out_mem, hidden_bytes, m_stream);
  net_launch(net_lm);

  int token = 0;
  tpuRtMemcpyD2SAsync((void *)&token, lm_out_mem, sizeof(int), m_stream);
  tpuRtStreamSynchronize(m_stream);

  visited_tokens[token_length] = token;
  token_length++;
  history_length++;
  return token;
}

void Qwen::clear_kv() {
  if (!support_prefill_kv) {
    return;
  }
  for (int i = 0; i < NUM_LAYERS; i++) {
    empty(past_key[i], kv_buffer_size, m_stream);
    empty(past_value[i], kv_buffer_size, m_stream);
  }
  history_length = 0;
}

PYBIND11_MODULE(chat, m) {
  py::class_<Qwen>(m, "Qwen")
      .def(py::init<>())
      .def("init", &Qwen::init)
      .def("forward_first", &Qwen::forward_first)
      .def("forward_next", &Qwen::forward_next)
      .def("clear_kv", &Qwen::clear_kv)
      .def("deinit", &Qwen::deinit)
      .def_readonly("SEQLEN", &Qwen::SEQLEN) // read SEQLEN in pipeline.py
      .def_readonly("MAX_INPUT_LENGTH", &Qwen::MAX_INPUT_LENGTH)
      .def_readonly("token_length", &Qwen::token_length)
      .def_readonly("history_length", &Qwen::history_length)
      .def_readonly("support_prefill_kv", &Qwen::support_prefill_kv)
      .def_readwrite("generation_mode", &Qwen::generation_mode)
      .def_readwrite("penalty", &Qwen::penalty)
      .def_readwrite("temperature", &Qwen::temperature)
      .def_readwrite("top_k", &Qwen::top_k)
      .def_readwrite("top_p", &Qwen::top_p);
}
