//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "bmruntime_interface.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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

void empty(bm_handle_t &bm_handle, bm_device_mem_t &mem) {
  int value = 0;
  auto ret = bm_memset_device_ext(bm_handle, &value, 1, mem);
  assert(BM_SUCCESS == ret);
}

// f32 → bf16 conversion (truncate lower 16 bits)
static void f32_to_bf16(const float *src, uint16_t *dst, int count) {
  const uint16_t *p = reinterpret_cast<const uint16_t *>(src);
  for (int i = 0; i < count; i++) {
    dst[i] = p[i * 2 + 1]; // upper 16 bits of float32
  }
}

// bf16 → f32 conversion
static float bf16_to_f32(uint16_t val) {
  uint32_t u = (uint32_t)val << 16;
  return *reinterpret_cast<float *>(&u);
}

class BgeReranker {
public:
  void init(const std::vector<int> &devid, std::string model_path);
  void deinit();
  float forward(std::vector<int> &input_ids, std::vector<int> &position_ids,
                std::vector<int> &attention_mask);

  int SEQLEN;
  int NUM_LAYERS;
  int HIDDEN_SIZE;

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);

  bm_handle_t bm_handle;
  void *p_bmrt;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_classifier;
  std::vector<const bm_net_info_t *> net_blocks;
  bm_device_mem_t bf16_buffer; // reusable bf16 buffer for type conversion
  int hidden_bytes_bf16;
  int hidden_bytes_f32;
};

void BgeReranker::net_launch(const bm_net_info_t *net, int stage_idx) {
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
}

void BgeReranker::init(const std::vector<int> &devices,
                        std::string model_path) {
  std::cout << "Device [ ";
  for (auto d : devices) {
    std::cout << d << " ";
  }
  std::cout << "] loading ....\n";

  bm_status_t status = bm_dev_request(&bm_handle, devices[0]);
  assert(BM_SUCCESS == status);

  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);

  std::cout << "Model [" << model_path.c_str() << "] loading .... ";
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  std::cout << "Done!" << std::endl;
  print_devmem_info(bm_handle);

  // Find embedding net
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1];

  // Find classifier net
  net_classifier = bmrt_get_network_info(p_bmrt, "classifier");

  // Find block nets
  const char **net_names = nullptr;
  auto num_nets = bmrt_get_network_number(p_bmrt);
  bmrt_get_network_names(p_bmrt, &net_names);

  auto is_exist = [](const char *name, const char **names, int num) {
    for (int i = 0; i < num; i++) {
      if (strcmp(name, names[i]) == 0)
        return true;
    }
    return false;
  };

  NUM_LAYERS = 0;
  for (int i = 0;; i++) {
    auto block_name = "block_" + std::to_string(i);
    if (!is_exist(block_name.c_str(), net_names, num_nets))
      break;
    net_blocks.emplace_back(
        bmrt_get_network_info(p_bmrt, block_name.c_str()));
    NUM_LAYERS++;
  }
  free(net_names);
  printf("Num Layers: %d, SEQLEN: %d\n", NUM_LAYERS, SEQLEN);

  auto &out_shape = net_blocks[0]->stages[0].output_shapes[0];
  HIDDEN_SIZE = out_shape.dims[2];
  printf("Hidden Size: %d\n", HIDDEN_SIZE);

  hidden_bytes_bf16 = SEQLEN * HIDDEN_SIZE * 2;
  hidden_bytes_f32 = SEQLEN * HIDDEN_SIZE * 4;

  // Allocate reusable bf16 buffer for type conversion
  bm_malloc_device_byte(bm_handle, &bf16_buffer, hidden_bytes_bf16);
}

void BgeReranker::deinit() {
  bm_free_device(bm_handle, bf16_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

float BgeReranker::forward(std::vector<int> &input_ids,
                            std::vector<int> &position_ids,
                            std::vector<int> &attention_mask) {
  int token_length = input_ids.size();
  assert(token_length <= SEQLEN);
  assert((int)position_ids.size() == SEQLEN);
  assert((int)attention_mask.size() == SEQLEN);

  // ====== Step 1: Embedding net ======
  // Inputs: input_ids [1, SEQLEN] int32, position_ids [1, SEQLEN] int32
  // Output: hidden_states [1, SEQLEN, HIDDEN_SIZE] f32
  auto &embed_in0 = net_embed->stages[0].input_mems[0];
  auto &embed_in1 = net_embed->stages[0].input_mems[1];
  auto &embed_out = net_embed->stages[0].output_mems[0];

  empty(bm_handle, embed_in0);
  empty(bm_handle, embed_in1);
  bm_memcpy_s2d_partial(bm_handle, embed_in0, input_ids.data(),
                         token_length * sizeof(int));
  bm_memcpy_s2d(bm_handle, embed_in1, position_ids.data());
  net_launch(net_embed);
  bm_thread_sync(bm_handle);

  // Read f32 output, convert to bf16, write to bf16 buffer
  std::vector<float> hidden_f32(SEQLEN * HIDDEN_SIZE);
  bm_memcpy_d2s(bm_handle, hidden_f32.data(), embed_out);

  std::vector<uint16_t> hidden_bf16(SEQLEN * HIDDEN_SIZE);
  f32_to_bf16(hidden_f32.data(), hidden_bf16.data(), SEQLEN * HIDDEN_SIZE);
  bm_memcpy_s2d(bm_handle, bf16_buffer, hidden_bf16.data());

  // ====== Step 2: Compute additive attention mask ======
  // Binary mask: 1 = real token, 0 = padding
  // Additive mask: 0 = real, -10000 = padding (bf16)
  // Block mask input shape: [1, 1, 1, SEQLEN]
  std::vector<uint16_t> additive_mask_bf16(SEQLEN);
  for (int i = 0; i < SEQLEN; i++) {
    if (attention_mask[i] == 0) {
      // -10000.0f in bf16
      float neg = -10000.0f;
      uint16_t tmp;
      f32_to_bf16(&neg, &tmp, 1);
      additive_mask_bf16[i] = tmp;
    } else {
      additive_mask_bf16[i] = 0;
    }
  }

  // ====== Step 3: Block nets (24 encoder layers) ======
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &block_in0 = net_blocks[idx]->stages[0].input_mems[0]; // hidden bf16
    auto &block_in1 = net_blocks[idx]->stages[0].input_mems[1]; // mask bf16

    // Copy bf16 hidden states to block input
    bm_memcpy_d2d_byte(bm_handle, block_in0, 0, bf16_buffer, 0,
                        hidden_bytes_bf16);

    // Copy additive mask to every block's input (each net has its own memory)
    bm_memcpy_s2d(bm_handle, block_in1, additive_mask_bf16.data());

    net_launch(net_blocks[idx]);
    bm_thread_sync(bm_handle);

    // Block output is f32 — read, convert to bf16 for next block/classifier
    auto &block_out = net_blocks[idx]->stages[0].output_mems[0];
    bm_memcpy_d2s(bm_handle, hidden_f32.data(), block_out);
    f32_to_bf16(hidden_f32.data(), hidden_bf16.data(), SEQLEN * HIDDEN_SIZE);
    bm_memcpy_s2d(bm_handle, bf16_buffer, hidden_bf16.data());
  }

  // ====== Step 4: Classifier net ======
  // Input: hidden_states [1, SEQLEN, HIDDEN_SIZE] bf16
  // Output: logit [1, 1] bf16
  auto &cls_in = net_classifier->stages[0].input_mems[0];
  bm_memcpy_d2d_byte(bm_handle, cls_in, 0, bf16_buffer, 0, hidden_bytes_bf16);
  net_launch(net_classifier);

  bm_thread_sync(bm_handle);

  // Read classifier output (bf16 [1,1])
  auto &cls_out = net_classifier->stages[0].output_mems[0];
  uint16_t logit_bf16;
  bm_memcpy_d2s_partial(bm_handle, &logit_bf16, cls_out, sizeof(uint16_t));
  float logit = bf16_to_f32(logit_bf16);

  return logit;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<BgeReranker>(m, "BgeReranker")
      .def(pybind11::init<>())
      .def("init", &BgeReranker::init)
      .def("deinit", &BgeReranker::deinit)
      .def("forward", &BgeReranker::forward)
      .def_readonly("SEQLEN", &BgeReranker::SEQLEN)
      .def_readonly("NUM_LAYERS", &BgeReranker::NUM_LAYERS)
      .def_readonly("HIDDEN_SIZE", &BgeReranker::HIDDEN_SIZE);
}
