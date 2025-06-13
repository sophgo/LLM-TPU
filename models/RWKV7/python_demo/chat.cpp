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
#include <vector>
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
#include <dlfcn.h>
#include "utils.h"

class RWKV7 {
public:
  void init(const std::vector<int> &devid, std::string model_path);
  void deinit();
  std::vector<float> forward_seq(std::vector<int> &tokens);
  std::vector<float> forward_one(int &token);
  void load_state(std::vector<float> &state);
  std::vector<float> clear_state();
  std::mt19937 sgen;
  RWKV7() : sgen(std::random_device()()){};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset = 0, int size = 0);

public:
  int token_length;
  int CHUNK_LEN;
  int NUM_LAYERS;
  std::vector<int> visited_tokens;

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  const bm_net_info_t *net_forward_seq, *net_forward_one;
  std::vector<bm_device_mem_t> state0;
  std::vector<bm_device_mem_t> state1;
  std::vector<bm_device_mem_t> state2;
};

void RWKV7::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset, int size) {
  if (!size) size = bm_mem_get_device_size(src);
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void RWKV7::init(const std::vector<int> &devices, std::string model_path) {
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
  bool ret = false;
  ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");

  // net embed and lm_head
  net_forward_one = bmrt_get_network_info(p_bmrt, "rwkv_forward_one");
  net_forward_seq = bmrt_get_network_info(p_bmrt, "rwkv_forward_seq");

  CHUNK_LEN = net_forward_seq->stages[0].input_shapes[0].dims[0];
  NUM_LAYERS = (net_forward_one->output_num - 1) / 3;
  assert(net_forward_seq->input_num == NUM_LAYERS * 3 + 1);
  assert(net_forward_seq->output_num == NUM_LAYERS * 3 + 1);

  visited_tokens.resize(CHUNK_LEN);
  state0.resize(NUM_LAYERS);
  state1.resize(NUM_LAYERS);
  state2.resize(NUM_LAYERS);

  int value = 0;
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_malloc_device_byte(bm_handle, &state0[i], 
                          net_forward_one->max_input_bytes[1]);
    bm_malloc_device_byte(bm_handle, &state1[i],
                          net_forward_one->max_input_bytes[1 + NUM_LAYERS]);
    bm_malloc_device_byte(bm_handle, &state2[i],
                          net_forward_one->max_input_bytes[1]);
    bm_memset_device_ext(bm_handle, &value, 1, state0[i]);
    bm_memset_device_ext(bm_handle, &value, 1, state1[i]);
    bm_memset_device_ext(bm_handle, &value, 1, state2[i]);
  }
}

void RWKV7::deinit() {
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_free_device(bm_handle, state0[i]);
    bm_free_device(bm_handle, state1[i]);
    bm_free_device(bm_handle, state2[i]);
  }
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
}

void RWKV7::net_launch(const bm_net_info_t *net, int stage_idx) {
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

std::vector<float> RWKV7::forward_seq(std::vector<int> &tokens) {
  auto &in_mem = net_forward_seq->stages[0].input_mems[0];
  auto &out_mem = net_forward_seq->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)tokens.data());
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_set_device_mem(
        &net_forward_seq->stages[0].input_mems[i+1],
        bm_mem_get_device_size(state0[i]),
        state0[i].u.device.device_addr);
    bm_set_device_mem(
        &net_forward_seq->stages[0].input_mems[i+1+NUM_LAYERS],
        bm_mem_get_device_size(state1[i]),
        state1[i].u.device.device_addr);
    bm_set_device_mem(
        &net_forward_seq->stages[0].input_mems[i+1+2*NUM_LAYERS],
        bm_mem_get_device_size(state2[i]),
        state2[i].u.device.device_addr);
    bm_set_device_mem(
        &net_forward_seq->stages[0].output_mems[i+1],
        bm_mem_get_device_size(state0[i]),
        state0[i].u.device.device_addr);
    bm_set_device_mem(
        &net_forward_seq->stages[0].output_mems[i+1+NUM_LAYERS],
        bm_mem_get_device_size(state1[i]),
        state1[i].u.device.device_addr);
    bm_set_device_mem(
        &net_forward_seq->stages[0].output_mems[i+1+2*NUM_LAYERS],
        bm_mem_get_device_size(state2[i]),
        state2[i].u.device.device_addr);  
  }
  net_launch(net_forward_seq);

  std::vector<float> logits(65536, 0.);
  bm_memcpy_d2s(bm_handle, (void *)logits.data(), out_mem);
  return logits;
}

std::vector<float> RWKV7::forward_one(int &token) {
  auto &in_mem = net_forward_one->stages[0].input_mems[0];
  auto &out_mem = net_forward_one->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)&token);
  for (int i = 0; i < NUM_LAYERS; i++) {
    bm_set_device_mem(
        &net_forward_one->stages[0].input_mems[i+1],
        bm_mem_get_device_size(state0[i]),
        state0[i].u.device.device_addr);
    bm_set_device_mem(
        &net_forward_one->stages[0].input_mems[i+1+NUM_LAYERS],
        bm_mem_get_device_size(state1[i]),
        state1[i].u.device.device_addr);
    bm_set_device_mem(
        &net_forward_one->stages[0].input_mems[i+1+2*NUM_LAYERS],
        bm_mem_get_device_size(state2[i]),
        state2[i].u.device.device_addr);
    bm_set_device_mem(
        &net_forward_one->stages[0].output_mems[i+1],
        bm_mem_get_device_size(state0[i]),
        state0[i].u.device.device_addr);
    bm_set_device_mem(
        &net_forward_one->stages[0].output_mems[i+1+NUM_LAYERS],
        bm_mem_get_device_size(state1[i]),
        state1[i].u.device.device_addr);
    bm_set_device_mem(
        &net_forward_one->stages[0].output_mems[i+1+2*NUM_LAYERS],
        bm_mem_get_device_size(state2[i]),
        state2[i].u.device.device_addr);  
  }
  net_launch(net_forward_one);

  std::vector<float> logits(65536, 0.);
  bm_memcpy_d2s(bm_handle, (void *)logits.data(), out_mem);
  return logits;
}

void RWKV7::load_state(std::vector<float> &state) {
  return;
}

std::vector<float> RWKV7::clear_state() {
  /*
  * init_ctx :
  *  "User: hi" + "\n\n" + 
  *  "Assistant: Hi. I am your assistant and I will provide expert full response in full details. 
  *   Please feel free to ask any question and I will always answer it." + "\n\n"
  */
  std::vector<int> init_tokens = {24281, 59, 4571, 261, 5585, 41693, 59, 3880, 47, 308, 4418, 32515, 59179, 21265, 308, 32475, 52597, 45929, 30923, 57119, 4596, 30923, 51454, 47, 44712, 30836, 30911, 4811, 21295, 21273, 57009, 21265, 308, 32475, 45150, 45175, 4601, 47, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 261};
  return forward_seq(init_tokens);
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<RWKV7>(m, "RWKV7")
      .def(pybind11::init<>())
      .def("init", &RWKV7::init)
      .def("deinit", &RWKV7::deinit)
      .def("forward_one", &RWKV7::forward_one)
      .def("forward_seq", &RWKV7::forward_seq)
      .def("load_state", &RWKV7::load_state)
      .def("clear_state", &RWKV7::clear_state)
      .def_readwrite("NUM_LAYERS", &RWKV7::NUM_LAYERS)
      .def_readwrite("CHUNK_LEN", &RWKV7::CHUNK_LEN);
}
