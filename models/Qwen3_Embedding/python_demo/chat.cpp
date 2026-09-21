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
#include <inttypes.h>
#include <iostream>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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

void empty(bm_handle_t &bm_handle, bm_device_mem_t &mem) {
  int value = 0;
  auto ret = bm_memset_device_ext(bm_handle, &value, 1, mem);
  assert(BM_SUCCESS == ret);
}

void empty_net(bm_handle_t &bm_handle, const bm_net_info_t *net,
               int stage_idx = 0) {
  int value = 0;
  for (int i = 0; i < net->input_num; i++) {
    bm_memset_device_ext(bm_handle, &value, 1,
                         net->stages[stage_idx].input_mems[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bm_memset_device_ext(bm_handle, &value, 1,
                         net->stages[stage_idx].output_mems[i]);
  }
}

class Qwen3Embedding {
public:
  void init(const std::vector<int> &devid, std::string model_path);
  void deinit();
  std::vector<float> forward(std::vector<int> &tokens);

  int SEQLEN;
  int NUM_LAYERS;
  int HIDDEN_SIZE;
  bool is_dynamic;

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  void net_launch_dyn(const bm_net_info_t *net, int real_len,
                      int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset = 0,
                  int size = 0);
  std::vector<float> read_hidden(bm_device_mem_t &mem, int num_tokens);

  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  const bm_net_info_t *net_embed;
  bm_device_mem_t dev_buffer;
  int hidden_bytes;
  bool prefill_mask;
  uint16_t mask_value;
  bool is_same_addr;
};

void Qwen3Embedding::d2d(bm_device_mem_t &dst, bm_device_mem_t &src,
                          int offset, int size) {
  if (!size)
    size = bm_mem_get_device_size(src);
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void Qwen3Embedding::net_launch(const bm_net_info_t *net, int stage_idx) {
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

void Qwen3Embedding::net_launch_dyn(const bm_net_info_t *net, int real_len,
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

  in_tensors[0].shape.dims[1] = real_len;
  in_tensors[1].shape.dims[1] = real_len;
  if (prefill_mask) {
    in_tensors[2].shape.dims[2] = real_len;
    in_tensors[2].shape.dims[3] = real_len;
  }

  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
}

void Qwen3Embedding::init(const std::vector<int> &devices,
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

  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  SEQLEN = net_embed->stages[0].input_shapes[0].dims[1];

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

  is_dynamic = net_blocks[0]->is_dynamic;
  prefill_mask = net_blocks[0]->input_num > 2;

  auto &out0_shape = net_blocks[0]->stages[0].output_shapes[0];
  HIDDEN_SIZE = out0_shape.dims[2];
  printf("Hidden Size: %d, Dynamic: %s\n", HIDDEN_SIZE,
         is_dynamic ? "true" : "false");

  if (net_blocks[0]->output_dtypes[0] == BM_FLOAT16) {
    mask_value = 0xF0E2;
    hidden_bytes = HIDDEN_SIZE * 2;
  } else if (net_blocks[0]->output_dtypes[0] == BM_BFLOAT16) {
    mask_value = 0xC61C;
    hidden_bytes = HIDDEN_SIZE * 2;
  } else {
    mask_value = 0;
    hidden_bytes = HIDDEN_SIZE * 4;
  }

  auto buffer_size =
      bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);

  is_same_addr = false;
  if (net_blocks[0]->stages[0].input_mems[0].u.device.device_addr ==
      net_blocks[0]->stages[0].output_mems[0].u.device.device_addr) {
    is_same_addr = true;
  }
}

void Qwen3Embedding::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

std::vector<float> Qwen3Embedding::read_hidden(bm_device_mem_t &mem,
                                                int num_tokens) {
  int out_bytes = num_tokens * hidden_bytes;
  std::vector<uint16_t> raw(num_tokens * HIDDEN_SIZE);
  bm_memcpy_d2s_partial(bm_handle, raw.data(), mem, out_bytes);

  bool is_bf16 = (net_blocks[0]->output_dtypes[0] == BM_BFLOAT16);
  std::vector<float> result(num_tokens * HIDDEN_SIZE);
  for (int i = 0; i < num_tokens * HIDDEN_SIZE; i++) {
    if (is_bf16) {
      uint32_t val = (uint32_t)raw[i] << 16;
      result[i] = *reinterpret_cast<float *>(&val);
    } else {
      uint16_t h = raw[i];
      uint32_t sign = (h >> 15) & 1;
      uint32_t exp = (h >> 10) & 0x1f;
      uint32_t mant = h & 0x3ff;
      uint32_t f;
      if (exp == 0) {
        if (mant == 0) {
          f = sign << 31;
        } else {
          uint32_t e = 1;
          while (!(mant & 0x400)) {
            mant <<= 1;
            e--;
          }
          mant &= 0x3ff;
          f = (sign << 31) | ((e + 127 - 15) << 23) | (mant << 13);
        }
      } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000 | (mant << 13);
      } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
      }
      result[i] = *reinterpret_cast<float *>(&f);
    }
  }
  return result;
}

std::vector<float> Qwen3Embedding::forward(std::vector<int> &tokens) {
  int token_length = tokens.size();
  assert(token_length <= SEQLEN);

  std::vector<int> position_id(SEQLEN, 0);
  for (int i = 0; i < token_length; i++) {
    position_id[i] = i;
  }

  std::vector<uint16_t> attn_mask;
  if (prefill_mask) {
    int dim = is_dynamic ? token_length : SEQLEN;
    attn_mask.resize(dim * dim, mask_value);
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attn_mask[i * dim + j] = 0;
      }
    }
  }

  auto in_mem = net_embed->stages[0].input_mems[0];
  auto out_mem = net_embed->stages[0].output_mems[0];
  empty(bm_handle, in_mem);
  bm_memcpy_s2d_partial(bm_handle, in_mem, (void *)tokens.data(),
                        token_length * sizeof(int));
  net_launch(net_embed);
  d2d(dev_buffer, out_mem, 0, bm_mem_get_device_size(out_mem));
  out_mem = dev_buffer;

  empty_net(bm_handle, net_blocks[0]);
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    if (!is_same_addr || idx == 0) {
      d2d(in0_mem, out_mem, 0, token_length * hidden_bytes);
    }
    if (idx == 0) {
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
      if (prefill_mask) {
        auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attn_mask.data());
      }
    }
    if (is_dynamic) {
      net_launch_dyn(net_blocks[idx], token_length);
    } else {
      net_launch(net_blocks[idx]);
    }
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
  }
  bm_thread_sync(bm_handle);

  return read_hidden(out_mem, token_length);
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Qwen3Embedding>(m, "Qwen3Embedding")
      .def(pybind11::init<>())
      .def("init", &Qwen3Embedding::init)
      .def("deinit", &Qwen3Embedding::deinit)
      .def("forward", &Qwen3Embedding::forward)
      .def_readonly("SEQLEN", &Qwen3Embedding::SEQLEN)
      .def_readonly("NUM_LAYERS", &Qwen3Embedding::NUM_LAYERS)
      .def_readonly("HIDDEN_SIZE", &Qwen3Embedding::HIDDEN_SIZE)
      .def_readonly("is_dynamic", &Qwen3Embedding::is_dynamic);
}
