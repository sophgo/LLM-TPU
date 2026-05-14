//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "embed_vit.hpp"
#include <fstream>
#include <iostream>

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

//===------------------------------------------------------------===//
// EmbedVit
//===------------------------------------------------------------===//

void EmbedVit::init_by_names() {

  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_embed_cache = bmrt_get_network_info(p_bmrt, "embedding_cache");
  net_vit = bmrt_get_network_info(p_bmrt, "vit");
  vit_dynamic = net_vit->is_dynamic;

  MAX_INPUT_LENGTH = net_embed->stages[0].input_shapes[0].dims[1];
  HIDDEN_SIZE = net_embed->stages[0].output_shapes[0].dims[2];
  SEQLEN = MAX_INPUT_LENGTH;
  MAX_PATCHES = net_vit->stages[0].input_shapes[0].dims[0];
  MAX_PIXELS = MAX_PATCHES * 16 * 16;
  VIT_DIMS = net_vit->stages[0].input_shapes[0].dims[1];
  printf("Max Pixels: %d*%d*%d\n", MAX_PATCHES / 4, 32, 32);
  num_deepstack = net_vit->output_num - 1;
}

void EmbedVit::init(int dev_id, std::string model_path) {
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

  auto buffer_size =
      bm_mem_get_device_size(net_embed->stages[0].output_mems[0]);
  status = bm_malloc_device_byte(bm_handle, &dev_buffer, buffer_size);
  assert(BM_SUCCESS == status);
  for (int i = 0; i < num_deepstack; i++) {
    bm_device_mem_t mem;
    status = bm_malloc_device_byte(bm_handle, &mem, buffer_size);
    assert(BM_SUCCESS == status);
    deepstack_buffers.push_back(mem);
  }
  vit_run = false;
}

void EmbedVit::deinit() {
  for (int i = 0; i < num_deepstack; i++) {
    bm_free_device(bm_handle, deepstack_buffers[i]);
  }
  bm_free_device(bm_handle, dev_buffer);
  bmrt_destroy(p_bmrt);
  bm_dev_free(bm_handle);
}

void EmbedVit::forward_embed(ArrayInt const &tokens) {
  std::fill(visited_tokens.begin(), visited_tokens.end(), 0);
  std::copy(tokens.begin(), tokens.end(), visited_tokens.data());
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_embed, in_tensors, out_tensors);
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)visited_tokens.data(),
                        MAX_INPUT_LENGTH * sizeof(int));
  net_launch(p_bmrt, net_embed, in_tensors, out_tensors);
  empty(bm_handle, dev_buffer);
  d2d(bm_handle, dev_buffer, out_tensors[0].device_mem, 0,
      tokens.size() * HIDDEN_SIZE * sizeof(uint16_t));
  token_length = tokens.size();
  for (auto &mem : deepstack_buffers) {
    empty(bm_handle, mem);
  }
}

void EmbedVit::forward_vit(const float *pixel_values,
                           ArrayInt const &position_ids,
                           ArrayInt const &pos_idx,
                           ArrayFloat const &pos_weight,
                           ArrayInt const &grid_thw, int vit_offset) {
  const int *p_thw = grid_thw.data();
  int t = p_thw[0];
  int h = p_thw[1];
  int w = p_thw[2];
  int hw = t * h * w;
  int num_pixels = hw * VIT_DIMS;
  assert((int)position_ids.size() == (hw * 2));
  assert((int)pos_idx.size() == 4 * hw);
  assert((int)pos_weight.size() == 4 * hw);
  auto p_position_ids = position_ids.data();
  auto p_pos_idx = pos_idx.data();
  auto p_pos_weight = pos_weight.data();
  empty_net(bm_handle, net_vit);
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_vit, in_tensors, out_tensors);
  bm_memcpy_s2d_partial(bm_handle, in_tensors[0].device_mem,
                        (void *)pixel_values, num_pixels * sizeof(float));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[1].device_mem,
                        (void *)p_position_ids,
                        position_ids.size() * sizeof(int));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[2].device_mem, (void *)p_pos_idx,
                        pos_idx.size() * sizeof(int));
  bm_memcpy_s2d_partial(bm_handle, in_tensors[3].device_mem,
                        (void *)p_pos_weight,
                        pos_weight.size() * sizeof(float));
  if (vit_dynamic) {
    in_tensors[0].shape.dims[0] = hw;
    in_tensors[1].shape.dims[0] = hw;
    in_tensors[2].shape.dims[0] = hw;
    in_tensors[3].shape.dims[0] = hw;
  } else {
    std::vector<float> attention_mask(MAX_PATCHES * MAX_PATCHES, -10000.0f);
    for (int i = 0; i < hw; i++) {
      auto row_begin = attention_mask.begin() + i * MAX_PATCHES;
      std::fill(row_begin, row_begin + hw, 0.0f);
    }
    bm_memcpy_s2d(bm_handle, in_tensors[4].device_mem,
                  (void *)attention_mask.data());
  }

  net_launch(p_bmrt, net_vit, in_tensors, out_tensors);

  // concatenate text embedding and image embedding
  int dst_offset = vit_offset * HIDDEN_SIZE * sizeof(uint16_t);
  int vit_size = hw / 4 * HIDDEN_SIZE * sizeof(uint16_t);
  bm_memcpy_d2d_byte(bm_handle, dev_buffer, dst_offset,
                     out_tensors[0].device_mem, 0, vit_size);
  for (int i = 0; i < num_deepstack; i++) {
    bm_memcpy_d2d_byte(bm_handle, deepstack_buffers[i], dst_offset,
                       out_tensors[i + 1].device_mem, 0, vit_size);
  }
  vit_run = true;
}

ArrayUint16 EmbedVit::forward_embed_cache(int token) {
  assert(net_embed_cache != nullptr &&
         "embedding_cache network not found in embed_vit.bmodel");
  std::vector<bm_tensor_t> in_tensors;
  std::vector<bm_tensor_t> out_tensors;
  init_tensors(net_embed_cache, in_tensors, out_tensors);
  bm_memcpy_s2d(bm_handle, in_tensors[0].device_mem, (void *)&token);
  net_launch(p_bmrt, net_embed_cache, in_tensors, out_tensors);
  ArrayUint16 result(HIDDEN_SIZE);
  bm_memcpy_d2s_partial(bm_handle, result.data(), out_tensors[0].device_mem,
                        HIDDEN_SIZE * sizeof(uint16_t));
  return result;
}

ArrayUint16 EmbedVit::get_hidden_states() {
  int bytes = token_length * HIDDEN_SIZE * sizeof(uint16_t);
  ArrayUint16 result(token_length * HIDDEN_SIZE);
  bm_memcpy_d2s_partial(bm_handle, result.data(), dev_buffer, bytes);
  return result;
}

ArrayUint162D EmbedVit::get_deepstacks() {
  ArrayUint162D result;
  int bytes = token_length * HIDDEN_SIZE * sizeof(uint16_t);
  for (int i = 0; i < num_deepstack; i++) {
    ArrayUint16 ds(token_length * HIDDEN_SIZE);
    bm_memcpy_d2s_partial(bm_handle, ds.data(), deepstack_buffers[i], bytes);
    result.push_back(std::move(ds));
  }
  return result;
}
