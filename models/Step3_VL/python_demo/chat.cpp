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
#include <cstdlib>
#include <getopt.h>
#include <inttypes.h>
#include <iostream>
#include <numeric>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
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

namespace py = pybind11;
using ArrayFloat =
    py::array_t<float, py::array::c_style | py::array::forcecast>;
using ArrayInt = py::array_t<int, py::array::c_style | py::array::forcecast>;

//===------------------------------------------------------------===//
// Empty Func
//===------------------------------------------------------------===//
void empty(bm_handle_t &bm_handle, bm_device_mem_t &mem) {
  int value = 0;
  auto ret = bm_memset_device_ext(bm_handle, &value, 1, mem);
  assert(BM_SUCCESS == ret);
}

void empty_net(bm_handle_t &bm_handle, const bm_net_info_t *net,
               int stage = 0) {
  for (int i = 0; i < net->input_num; i++) {
    empty(bm_handle, net->stages[stage].input_mems[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    empty(bm_handle, net->stages[stage].output_mems[i]);
  }
}

class Step3VL {
public:
  void init(int devid, std::string model_path);
  void deinit();
  void forward_embed(ArrayInt const &tokens);
  void forward_vit_global(ArrayFloat const &pixel_values, int vit_offset);
  void forward_vit_patch(ArrayFloat const &pixel_values, int patch_index,
                         int vit_offset);
  int forward_first(ArrayInt const &position_ids);
  int forward_next(ArrayInt const &position_ids);
  void clear_history();

  std::mt19937 sgen;
  Step3VL() : sgen(std::random_device()()){};

private:
  void net_launch(const bm_net_info_t *net,
                  const std::vector<bm_tensor_t> &in_tensors,
                  std::vector<bm_tensor_t> &out_tensors);
  void net_launch_decode(int block_idx, int kv_offset,
                         bm_device_mem_t &input_mem, const int *position_id,
                         std::vector<uint16_t> &attention_mask,
                         int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset = 0,
                  int size = 0);
  void init_by_names();
  int select_decode_stage();
  int generate(bm_device_mem_t &logits_mem);
  void init_tensors(const bm_net_info_t *net,
                    std::vector<bm_tensor_t> &in_tensors,
                    std::vector<bm_tensor_t> &out_tensors, int stage = 0);

public:
  int token_length;
  int history_length;
  int SEQLEN;
  int MAX_INPUT_LENGTH;
  int HIDDEN_SIZE;
  int KV_BYTES;
  int NUM_LAYERS;
  int MAX_PATCHES;
  int MAX_PIXELS;
  int GLOBAL_TOKENS;  // 13*13=169
  int PATCH_TOKENS;   // 9*9=81
  bool lmhead_with_topk;
  bool support_history;
  bool has_vit_patch;
  bool prefill_mask;
  uint16_t mask_value;
  std::vector<int> visited_tokens;

private:
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  std::vector<int> decode_stage_len;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_embed_cache;
  const bm_net_info_t *net_lm;
  const bm_net_info_t *net_vit_global;
  const bm_net_info_t *net_vit_patch;
  bm_device_mem_t dev_buffer;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};

void Step3VL::init_tensors(const bm_net_info_t *net,
                           std::vector<bm_tensor_t> &in_tensors,
                           std::vector<bm_tensor_t> &out_tensors, int stage) {
  in_tensors.resize(net->input_num);
  out_tensors.resize(net->output_num);
  for (int i = 0; i < net->input_num; i++) {
    bmrt_tensor_with_device(&in_tensors[i], net->stages[stage].input_mems[i],
                            net->input_dtypes[i],
                            net->stages[stage].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(&out_tensors[i], net->stages[stage].output_mems[i],
                            net->output_dtypes[i],
                            net->stages[stage].output_shapes[i]);
  }
}

void Step3VL::net_launch(const bm_net_info_t *net,
                         const std::vector<bm_tensor_t> &in_tensors,
                         std::vector<bm_tensor_t> &out_tensors) {
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
}

void Step3VL::net_launch_decode(int idx, int kv_offset,
                                bm_device_mem_t &input_mem, const int *pos_id,
                                std::vector<uint16_t> &attention_mask,
                                int stage_idx) {
  auto &net = net_blocks_cache[idx];
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net, in_tensors, out_tensors, stage_idx);

  // ===== prepare input tensors =====
  in_tensors[0].device_mem = input_mem;
  if (idx == 0) {
    bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem, (void *)pos_id);
    bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                  (void *)attention_mask.data());
  } else {
    in_tensors[1].device_mem =
        net_blocks_cache[0]->stages[stage_idx].input_mems[1];
    in_tensors[2].device_mem =
        net_blocks_cache[0]->stages[stage_idx].input_mems[2];
  }
  // The real KV lives in past_key/past_value; rebind history inputs to them
  // (only stage 0's input_mems is aliased there).
  int stage_capacity = net->stages[stage_idx].input_shapes[3].dims[1];
  in_tensors[3].device_mem = bm_mem_from_device(
      past_key[idx].u.device.device_addr, stage_capacity * KV_BYTES);
  in_tensors[4].device_mem = bm_mem_from_device(
      past_value[idx].u.device.device_addr, stage_capacity * KV_BYTES);
  out_tensors[1].device_mem = bm_mem_from_device(
      past_key[idx].u.device.device_addr + kv_offset, KV_BYTES);
  out_tensors[2].device_mem = bm_mem_from_device(
      past_value[idx].u.device.device_addr + kv_offset, KV_BYTES);

  // ===== launch =====
  net_launch(net, in_tensors, out_tensors);
}

void Step3VL::d2d(bm_device_mem_t &dst, bm_device_mem_t &src, int offset,
                  int size) {
  if (!size) {
    size = bm_mem_get_device_size(src);
  }
  bm_memcpy_d2d_byte(bm_handle, dst, offset, src, 0, size);
}

void Step3VL::clear_history() {
  // no history support in static mode without --use_history_kv
}

void Step3VL::init_by_names() {
  auto is_exist = [](const char *name, const char **names, int num) {
    for (int i = 0; i < num; i++) {
      if (strcmp(name, names[i]) == 0) {
        return true;
      }
    }
    return false;
  };
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_vit_global = bmrt_get_network_info(p_bmrt, "vit_global");
  net_lm = bmrt_get_network_info(p_bmrt, "lm_head");
  const char **net_names = nullptr;
  auto num_nets = bmrt_get_network_number(p_bmrt);
  bmrt_get_network_names(p_bmrt, &net_names);

  // count non-block networks: embed, embed_cache, lm_head, vit_global,
  // vit_patch (optional)
  int num_vit = 0;
  if (is_exist("vit_global", net_names, num_nets))
    num_vit++;
  has_vit_patch = is_exist("vit_patch", net_names, num_nets);
  if (has_vit_patch) {
    net_vit_patch = bmrt_get_network_info(p_bmrt, "vit_patch");
    num_vit++;
  } else {
    net_vit_patch = nullptr;
  }
  auto num_blocks = num_nets - 3 - num_vit; // 3 = embed + embed_cache + lm_head

  std::string kv_name = "block_kv_0";
  if (is_exist(kv_name.c_str(), net_names, num_nets)) {
    support_history = true;
  } else {
    support_history = false;
  }

  if (support_history) {
    NUM_LAYERS = num_blocks / 3;
  } else {
    NUM_LAYERS = num_blocks / 2;
  }

  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    if ((!is_exist(block_name.c_str(), net_names, num_nets)) ||
        (!is_exist(cache_name.c_str(), net_names, num_nets))) {
      NUM_LAYERS = i;
      printf("Warning: Only %d blocks found, total %d blocks.\n", NUM_LAYERS,
             num_blocks);
      break;
    }
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    auto cache_net = bmrt_get_network_info(p_bmrt, cache_name.c_str());
    net_blocks_cache.emplace_back(cache_net);
    // collect per-stage KV length from block_cache input[3]
    auto decode_stage_num = cache_net->stage_num;
    if (decode_stage_len.empty()) {
      for (int j = 0; j < decode_stage_num; j++) {
        decode_stage_len.push_back(
            cache_net->stages[j].input_shapes[3].dims[1]);
      }
    } else {
      assert(decode_stage_num == (int)decode_stage_len.size());
    }
  }
  free(net_names);

  if (net_embed_cache->output_dtypes[0] == BM_FLOAT16) {
    mask_value = 0xF0E2; // float16
  } else if (net_embed_cache->output_dtypes[0] == BM_BFLOAT16) {
    mask_value = 0xC61C; // -9984 by bfloat16
  } else {
    std::cerr << "\nError: Invalid attention dtype\n";
    throw std::runtime_error("Invalid attention dtype");
  }

  prefill_mask = net_blocks[0]->input_num > 2;
  history_length = 0;
  lmhead_with_topk = net_lm->stages[0].output_shapes[0].dims[1] == 1;
  MAX_INPUT_LENGTH = net_embed->stages[0].input_shapes[0].dims[1];
  HIDDEN_SIZE = net_lm->stages[0].input_shapes[0].dims[1];
  SEQLEN = net_blocks_cache[0]->stages[0].input_shapes[3].dims[1];
  KV_BYTES =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);

  // ViT dimensions
  GLOBAL_TOKENS = net_vit_global->stages[0].output_shapes[0].dims[1]; // 169
  PATCH_TOKENS = has_vit_patch ? net_vit_patch->stages[0].output_shapes[0].dims[1] : 0;   // 81
  MAX_PATCHES = has_vit_patch ? net_vit_patch->stages[0].input_shapes[0].dims[0] : 0;     // 4 or 0
  MAX_PIXELS = 728 * 728; // global view resolution

  printf("Num Layers:%d\n", NUM_LAYERS);
  printf("MAX_INPUT_LENGTH: %d\n", MAX_INPUT_LENGTH);
  printf("SEQLEN: %d\n", SEQLEN);
  printf("Global tokens: %d, Patch tokens: %d, Max patches: %d\n",
         GLOBAL_TOKENS, PATCH_TOKENS, MAX_PATCHES);
  printf("History Support: %s\n", support_history ? "True" : "False");
}

void Step3VL::init(int dev_id, std::string model_path) {
  // request bm_handle
  std::cout << "Device [ " << dev_id << " ] loading .....\n";
  bm_status_t status = bm_dev_request(&bm_handle, dev_id);
  assert(BM_SUCCESS == status);

  // create bmruntime
  p_bmrt = bmrt_create(bm_handle);
  assert(NULL != p_bmrt);
  bmrt_set_flags(p_bmrt, BM_RUNTIME_SHARE_MEM);
  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  bm_thread_sync(bm_handle);
  printf("Done!\n");
  print_devmem_info(bm_handle);

  init_by_names();

  visited_tokens.resize(SEQLEN);

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; i++) {
    past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
    past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    empty(bm_handle, past_key[i]);
    empty(bm_handle, past_value[i]);
  }
  uint32_t buffer_size = bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  assert(BM_SUCCESS == status);
}

void Step3VL::deinit() {
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

void Step3VL::forward_embed(ArrayInt const &tokens) {
  token_length = tokens.size();
  assert(token_length <= MAX_INPUT_LENGTH);

  auto p_buffer = tokens.request();
  auto p_tokens = static_cast<int *>(p_buffer.ptr);
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  std::copy(p_tokens, p_tokens + token_length, visited_tokens.data());

  empty(bm_handle, dev_buffer);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_embed, in_tensors, out_tensors);
  if (token_length != MAX_INPUT_LENGTH) {
    empty(bm_handle, in_tensors[0].device_mem);
  }
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)p_tokens, token_length * sizeof(int));
  net_launch(net_embed, in_tensors, out_tensors);
  d2d(dev_buffer, out_tensors[0].device_mem, 0,
      token_length * HIDDEN_SIZE * sizeof(uint16_t));
}

void Step3VL::forward_vit_global(ArrayFloat const &pixel_values,
                                 int vit_offset) {
  int num_pixels = 1 * 3 * 728 * 728;
  assert(pixel_values.size() == num_pixels);
  auto p_pixel_values = pixel_values.request();

  empty_net(bm_handle, net_vit_global);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_vit_global, in_tensors, out_tensors);
  bm_memcpy_s2d(bm_handle, in_tensors[0].device_mem,
                (void *)p_pixel_values.ptr);
  net_launch(net_vit_global, in_tensors, out_tensors);

  // write ViT output into embedding buffer at vit_offset
  int dst_offset = vit_offset * HIDDEN_SIZE * sizeof(uint16_t);
  int vit_size = GLOBAL_TOKENS * HIDDEN_SIZE * sizeof(uint16_t);
  bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset,
                     out_tensors[0].device_mem, 0, vit_size);
}

void Step3VL::forward_vit_patch(ArrayFloat const &pixel_values,
                                int patch_index, int vit_offset) {
  if (!has_vit_patch) {
    std::cerr << "Warning: vit_patch not compiled (max_patches=0), "
              << "skipping patch " << patch_index << std::endl;
    return;
  }
  // pixel_values contains ONE patch: [1, 3, 504, 504]
  int num_pixels = 1 * 3 * 504 * 504;
  assert(pixel_values.size() == num_pixels);
  auto p_pixel_values = pixel_values.request();

  empty_net(bm_handle, net_vit_patch);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_vit_patch, in_tensors, out_tensors);
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)p_pixel_values.ptr,
                        num_pixels * sizeof(float));
  net_launch(net_vit_patch, in_tensors, out_tensors);

  // Write single patch output (81 tokens) at vit_offset
  int dst_offset = vit_offset * HIDDEN_SIZE * sizeof(uint16_t);
  int vit_size = PATCH_TOKENS * HIDDEN_SIZE * sizeof(uint16_t);
  bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset,
                     out_tensors[0].device_mem, 0, vit_size);
}

int Step3VL::generate(bm_device_mem_t &logits_mem) {
  int token = 0;
  if (lmhead_with_topk) {
    bm_memcpy_d2s_partial(bm_handle, (void *)&token, logits_mem, sizeof(int));
  } else {
    // fallback: argmax on CPU
    std::vector<float> logits(HIDDEN_SIZE);
    bm_memcpy_d2s(bm_handle, logits.data(), logits_mem);
    token = std::distance(logits.begin(),
                          std::max_element(logits.begin(), logits.end()));
  }
  return token;
}

int Step3VL::forward_first(ArrayInt const &position_ids) {
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  assert((int)position_ids.size() == token_length);

  std::vector<int> position_ids_pad(MAX_INPUT_LENGTH, 0);
  std::copy(p_ids, p_ids + token_length, position_ids_pad.begin());

  std::vector<uint16_t> attention_mask;
  if (prefill_mask) {
    attention_mask.assign(MAX_INPUT_LENGTH * MAX_INPUT_LENGTH, mask_value);
    for (int i = 0; i < token_length; i++) {
      for (int j = 0; j <= i; j++) {
        attention_mask[i * MAX_INPUT_LENGTH + j] = 0;
      }
    }
  }

  auto out_mem = dev_buffer;
  empty_net(bm_handle, net_blocks[0]);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    init_tensors(net_blocks[idx], in_tensors, out_tensors);
    in_tensors[0].device_mem = out_mem;
    if (idx == 0) {
      bm_memcpy_s2d(bm_handle, in_tensors[1].device_mem,
                    (void *)position_ids_pad.data());
      if (prefill_mask) {
        bm_memcpy_s2d(bm_handle, in_tensors[2].device_mem,
                      (void *)attention_mask.data());
      }
    }
    net_launch(net_blocks[idx], in_tensors, out_tensors);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], 0,
                       net_blocks[idx]->stages[0].output_mems[1], 0,
                       KV_BYTES * token_length);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], 0,
                       net_blocks[idx]->stages[0].output_mems[2], 0,
                       KV_BYTES * token_length);
  }

  // forward lmhead
  int bytes = HIDDEN_SIZE * sizeof(uint16_t);
  init_tensors(net_lm, in_tensors, out_tensors);
  in_tensors[0].device_mem = bm_mem_from_device(
      out_mem.u.device.device_addr + (token_length - 1) * bytes, bytes);
  out_tensors[0].device_mem = dev_buffer;
  net_launch(net_lm, in_tensors, out_tensors);
  auto token = generate(dev_buffer);
  visited_tokens[token_length] = token;
  token_length++;
  history_length = token_length;
  return token;
}

int Step3VL::select_decode_stage() {
  if (decode_stage_len.empty()) {
    return 0;
  }
  int stage_idx = 0;
  for (auto &len : decode_stage_len) {
    if (history_length > len) {
      break;
    }
    stage_idx++;
  }
  if (stage_idx > 0) {
    stage_idx--;
  }
  return stage_idx;
}

int Step3VL::forward_next(ArrayInt const &position_ids) {
  int stage = select_decode_stage();
  int real_len = decode_stage_len.empty() ? SEQLEN : decode_stage_len[stage];
  std::vector<uint16_t> attention_mask(real_len + 1, 0);
  for (int i = history_length - 1; i < real_len; i++) {
    attention_mask[i] = mask_value;
  }
  assert(position_ids.size() == 1);
  auto p_position_ids = position_ids.request();
  auto p_ids = static_cast<int *>(p_position_ids.ptr);
  // embedding
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_embed_cache, in_tensors, out_tensors);
  int token = visited_tokens[token_length - 1];
  bm_memcpy_s2d(bm_handle, in_tensors[0].device_mem, (void *)&token);
  net_launch(net_embed_cache, in_tensors, out_tensors);
  auto out_mem = out_tensors[0].device_mem;

  // blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (history_length - 1) * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    net_launch_decode(idx, token_offset, out_mem, p_ids, attention_mask, stage);
    out_mem = net_blocks_cache[idx]->stages[stage].output_mems[0];
  }

  // forward lmhead
  init_tensors(net_lm, in_tensors, out_tensors);
  in_tensors[0].device_mem = out_mem;
  out_tensors[0].device_mem = dev_buffer;
  net_launch(net_lm, in_tensors, out_tensors);

  token = generate(dev_buffer);
  visited_tokens[token_length] = token;
  token_length++;
  history_length++;
  return token;
}

PYBIND11_MODULE(chat, m) {
  pybind11::class_<Step3VL>(m, "Step3VL")
      .def(pybind11::init<>())
      .def("init", &Step3VL::init)
      .def("forward_embed", &Step3VL::forward_embed)
      .def("forward_vit_global", &Step3VL::forward_vit_global)
      .def("forward_vit_patch", &Step3VL::forward_vit_patch)
      .def("forward_first", &Step3VL::forward_first)
      .def("forward_next", &Step3VL::forward_next)
      .def("clear_history", &Step3VL::clear_history)
      .def("deinit", &Step3VL::deinit)
      .def_readonly("SEQLEN", &Step3VL::SEQLEN)
      .def_readonly("MAX_INPUT_LENGTH", &Step3VL::MAX_INPUT_LENGTH)
      .def_readonly("MAX_PATCHES", &Step3VL::MAX_PATCHES)
      .def_readonly("MAX_PIXELS", &Step3VL::MAX_PIXELS)
      .def_readonly("GLOBAL_TOKENS", &Step3VL::GLOBAL_TOKENS)
      .def_readonly("PATCH_TOKENS", &Step3VL::PATCH_TOKENS)
      .def_readonly("support_history", &Step3VL::support_history)
      .def_readonly("has_vit_patch", &Step3VL::has_vit_patch)
      .def_readonly("history_length", &Step3VL::history_length);
}
